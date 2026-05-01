from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_tornado_ml_project as project


OUTPUT_DIR = ROOT / "outputs"
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_CSV = OUTPUT_DIR / "downsampled_xgboost_results.csv"
SUMMARY_JSON = OUTPUT_DIR / "downsampled_xgboost_summary.json"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def ensure_dirs() -> None:
    for path in [OUTPUT_DIR, ARTIFACTS_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


class RandomUnderSampleXGBClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"
    def __init__(
        self,
        *,
        random_state: int = 42,
        objective: str = "binary:logistic",
        eval_metric: str = "aucpr",
        tree_method: str = "hist",
        n_jobs: int = 1,
        verbosity: int = 0,
        n_estimators: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        min_child_weight: int = 1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
    ):
        self.random_state = random_state
        self.objective = objective
        self.eval_metric = eval_metric
        self.tree_method = tree_method
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

    def fit(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y).astype(int)
        positives = np.flatnonzero(y == 1)
        negatives = np.flatnonzero(y == 0)
        if len(positives) == 0 or len(negatives) == 0:
            raise ValueError("Both classes must be present for undersampling.")

        sample_n = min(len(positives), len(negatives))
        rng = np.random.default_rng(self.random_state)
        pos_keep = rng.choice(positives, size=sample_n, replace=False)
        neg_keep = rng.choice(negatives, size=sample_n, replace=False)
        keep_idx = np.concatenate([pos_keep, neg_keep])
        rng.shuffle(keep_idx)

        self.downsample_positive_count_ = int(sample_n)
        self.downsample_negative_count_ = int(sample_n)
        self.downsample_total_count_ = int(len(keep_idx))

        self.model_ = XGBClassifier(
            objective=self.objective,
            eval_metric=self.eval_metric,
            tree_method=self.tree_method,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=self.verbosity,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
        )
        self.model_.fit(x[keep_idx], y[keep_idx])
        self.classes_ = np.array([0, 1], dtype=int)
        self.n_features_in_ = x.shape[1]
        return self

    def predict_proba(self, x):
        return self.model_.predict_proba(np.asarray(x))

    def predict(self, x):
        scores = self.predict_proba(x)[:, 1]
        return (scores >= 0.5).astype(int)


def evaluate_split(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (scores >= threshold).astype(int)
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    auroc = roc_auc_score(y_true, scores)
    auprc = average_precision_score(y_true, scores)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def find_best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5
    f1_values = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.nanargmax(f1_values))
    return float(thresholds[best_idx])


def run_task(df: pd.DataFrame, task_name: str, target_col: str) -> dict[str, object]:
    split_idx = project.ensure_split_indices(df, task_name, target_col, force=False)
    selection_state = project.load_selection_state(task_name)
    spec = project.FEATURE_SPECS["county_enriched"]

    x_train = df.loc[split_idx["train"], spec.all_features]
    y_train = df.loc[split_idx["train"], target_col]
    x_val = df.loc[split_idx["val"], spec.all_features]
    y_val = df.loc[split_idx["val"], target_col].to_numpy()
    x_test = df.loc[split_idx["test"], spec.all_features]
    y_test = df.loc[split_idx["test"], target_col].to_numpy()

    pipeline = Pipeline(
        steps=[
            ("preprocess", project.build_preprocessor(spec, categories=selection_state["categorical_levels"])),
            (
                "selector",
                project.FixedFeatureSelector(
                    selected_indices=selection_state["selected_indices"],
                    selected_names=selection_state["selected_features"],
                ),
            ),
            (
                "classifier",
                RandomUnderSampleXGBClassifier(
                    random_state=project.RANDOM_STATE,
                    tree_method="hist",
                    n_jobs=project.TREE_WORKERS,
                    verbosity=0,
                ),
            ),
        ]
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions={
            "classifier__n_estimators": [200, 350],
            "classifier__max_depth": [3, 5, 7],
            "classifier__learning_rate": [0.03, 0.08, 0.15],
            "classifier__min_child_weight": [1, 3, 5],
            "classifier__subsample": [0.8, 1.0],
            "classifier__colsample_bytree": [0.7, 1.0],
        },
        n_iter=project.RANDOM_SEARCH_ITERS,
        random_state=project.RANDOM_STATE,
        cv=project.build_cv(project.DEFAULT_CV_SPLITS),
        scoring="average_precision",
        refit=True,
        n_jobs=1,
        error_score="raise",
    )

    log(f"Running downsampled XGBoost for {task_name}...")
    search.fit(x_train, y_train)
    best_model = search.best_estimator_

    val_scores = project.get_scores(best_model, x_val)
    threshold = find_best_f1_threshold(y_val, val_scores)
    val_metrics = evaluate_split(y_val, val_scores, threshold)

    test_scores = project.get_scores(best_model, x_test)
    test_metrics = evaluate_split(y_test, test_scores, threshold)

    classifier = best_model.named_steps["classifier"]
    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)

    artifact = {
        "task": task_name,
        "target_col": target_col,
        "dataset": str(project.INPUT_CSV),
        "feature_set": "county_selected",
        "model_name": "xgboost",
        "imbalance_strategy": "random_undersample_training_to_1to1_without_class_weights",
        "cv_splits": int(project.DEFAULT_CV_SPLITS),
        "scoring": "average_precision",
        "train_original_positive_count": pos_count,
        "train_original_negative_count": neg_count,
        "selected_feature_count": int(len(selection_state["selected_features"])),
        "selected_features": list(selection_state["selected_features"]),
        "best_params": search.best_params_,
        "cv_best_auprc": float(search.best_score_),
        "validation_threshold": float(threshold),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "undersampled_fit_positive_count": int(classifier.downsample_positive_count_),
        "undersampled_fit_negative_count": int(classifier.downsample_negative_count_),
        "undersampled_fit_total_count": int(classifier.downsample_total_count_),
        "notes": [
            "This isolated run keeps the previous reduced dataset, split indices, and selected feature set.",
            "Only the imbalance strategy changes: each fit downsamples the training portion to equal positives and negatives.",
            "Validation and test sets remain untouched and imbalanced.",
            "No class weights or scale_pos_weight are used anywhere in the XGBoost fit.",
        ],
    }

    artifact_path = ARTIFACTS_DIR / f"{task_name}_downsampled_xgboost.json"
    model_path = MODELS_DIR / f"{task_name}_downsampled_xgboost.joblib"
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    joblib.dump(best_model, model_path)

    return {
        "task": task_name,
        "feature_set": "county_selected",
        "model_name": "xgboost",
        "imbalance_strategy": artifact["imbalance_strategy"],
        "cv_splits": project.DEFAULT_CV_SPLITS,
        "cv_best_auprc": artifact["cv_best_auprc"],
        "validation_threshold": artifact["validation_threshold"],
        "selected_feature_count": artifact["selected_feature_count"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "val_auroc": val_metrics["auroc"],
        "val_auprc": val_metrics["auprc"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_auroc": test_metrics["auroc"],
        "test_auprc": test_metrics["auprc"],
        "train_original_positive_count": pos_count,
        "train_original_negative_count": neg_count,
        "undersampled_fit_positive_count": artifact["undersampled_fit_positive_count"],
        "undersampled_fit_negative_count": artifact["undersampled_fit_negative_count"],
        "undersampled_fit_total_count": artifact["undersampled_fit_total_count"],
        "best_params_json": json.dumps(search.best_params_, sort_keys=True),
        "artifact_path": str(artifact_path),
        "model_path": str(model_path),
    }


def main() -> int:
    ensure_dirs()
    df = project.load_dataset()
    rows = []
    for task_name, target_col in project.TASKS.items():
        rows.append(run_task(df, task_name, target_col))

    results = pd.DataFrame(rows).sort_values("task").reset_index(drop=True)
    results.to_csv(RESULTS_CSV, index=False)
    SUMMARY_JSON.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    log(f"Saved results to {RESULTS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
