#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "final_tornado_dataset_region_reduced.csv"
OUTPUT_DIR = ROOT / "analysis_outputs"
REPORTS_DIR = OUTPUT_DIR / "reports"
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"
MODELS_DIR = OUTPUT_DIR / "models"
SPLITS_DIR = OUTPUT_DIR / "splits"

RUN_TAG = INPUT_CSV.stem
RESULTS_CSV = REPORTS_DIR / f"tornado_ml_results__{RUN_TAG}.csv"
SUMMARY_MD = REPORTS_DIR / f"project_completion_summary__{RUN_TAG}.md"
METHODOLOGY_JSON = REPORTS_DIR / f"methodology_decisions__{RUN_TAG}.json"

RANDOM_STATE = 42
TRAIN_SHARE = 0.70
VAL_SHARE = 0.15
TEST_SHARE = 0.15

# User-approved pragmatic reduction from the PDF's 10-fold CV to finish within one day.
DEFAULT_CV_SPLITS = 3

BASELINE_C_GRID = [0.01, 0.1, 1.0, 5.0, 10.0]
SELECTOR_C = 0.05
SELECTOR_MAX_FEATURES = 20
NONZERO_EPS = 1e-10
RANDOM_SEARCH_ITERS = 6
SHAP_BACKGROUND_N = 80
SHAP_EVAL_N = 40
TREE_WORKERS = max(1, (os.cpu_count() or 2) - 1)

MODEL_ORDER = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "svm",
    "neural_network",
    "knn",
]

RESULT_COLUMNS = [
    "task",
    "experiment_group",
    "feature_set",
    "model_name",
    "cv_splits",
    "imbalance_strategy",
    "cv_best_auprc",
    "validation_threshold",
    "selected_feature_count",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_auroc",
    "val_auprc",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_auroc",
    "test_auprc",
]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    numeric_features: list[str]
    categorical_features: list[str]

    @property
    def all_features(self) -> list[str]:
        return self.numeric_features + self.categorical_features


class FixedFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, selected_indices: list[int], selected_names: list[str] | None = None):
        self.selected_indices = selected_indices
        self.selected_names = selected_names

    def fit(self, x, y=None):
        x = np.asarray(x)
        self.n_features_in_ = int(x.shape[1])
        selected_indices = [int(idx) for idx in self.selected_indices]
        if not selected_indices:
            raise ValueError("Feature selector received an empty selected_indices list.")
        if max(selected_indices) >= self.n_features_in_:
            raise ValueError("Feature selector indices exceed the transformed feature dimension.")
        self.selected_indices_ = selected_indices
        self.selected_names_ = list(self.selected_names or [])
        return self

    def transform(self, x):
        x = np.asarray(x)
        return x[:, self.selected_indices_]

    def get_support(self) -> np.ndarray:
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_indices_] = True
        return mask

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is None:
            if self.selected_names_:
                return np.asarray(self.selected_names_, dtype=object)
            return np.asarray([f"x{i}" for i in self.selected_indices_], dtype=object)
        return np.asarray([input_features[idx] for idx in self.selected_indices_], dtype=object)


def build_feature_specs() -> dict[str, FeatureSpec]:
    tornado_numeric = [
        "mag_num",
        "mag_missing",
        "len",
        "wid",
        "end_missing",
        "wid_missing",
        "len_zero",
    ]
    tornado_categorical = ["season", "region"]

    county_numeric = [
        "start_county_population",
        "start_county_population_density_km2",
        "start_county_mobile_home_share",
        "start_county_median_hh_income",
        "start_county_unemployment_rate",
        "start_county_disability_rate",
        "start_county_age_under18_share",
        "start_county_age_65plus_share",
        "end_county_population",
        "end_county_population_density_km2",
        "end_county_mobile_home_share",
        "end_county_median_hh_income",
        "end_county_unemployment_rate",
        "end_county_disability_rate",
        "end_county_age_under18_share",
        "end_county_age_65plus_share",
        "path_counties_n",
        "path_overlap_area_km2",
        "path_est_exposed_pop",
        "path_population_density_wavg",
        "path_mobile_home_share_wavg",
        "path_median_hh_income_wavg",
        "path_unemployment_rate_wavg",
        "path_disability_rate_wavg",
        "path_age_under18_share_wavg",
        "path_age_65plus_share_wavg",
    ]
    county_categorical: list[str] = []

    return {
        "tornado_only": FeatureSpec(
            name="tornado_only",
            numeric_features=tornado_numeric,
            categorical_features=tornado_categorical,
        ),
        "county_enriched": FeatureSpec(
            name="county_enriched",
            numeric_features=tornado_numeric + county_numeric,
            categorical_features=tornado_categorical + county_categorical,
        ),
    }


FEATURE_SPECS = build_feature_specs()
TASKS = {"injury": "inj_bin", "fatality": "fat_bin"}


def ensure_output_dirs() -> None:
    for path in [OUTPUT_DIR, REPORTS_DIR, ARTIFACTS_DIR, MODELS_DIR, SPLITS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def results_template() -> pd.DataFrame:
    return pd.DataFrame(columns=RESULT_COLUMNS)


def load_results() -> pd.DataFrame:
    if RESULTS_CSV.exists():
        return pd.read_csv(RESULTS_CSV)
    return results_template()


def save_results(df: pd.DataFrame) -> None:
    ordered = df.copy()
    if ordered.empty:
        ordered = results_template()
    ordered = ordered[RESULT_COLUMNS]
    ordered.to_csv(RESULTS_CSV, index=False)


def upsert_result(row: dict[str, object]) -> None:
    df = load_results()
    if df.empty:
        df = results_template()
    mask = (
        (df["task"] == row["task"])
        & (df["experiment_group"] == row["experiment_group"])
        & (df["feature_set"] == row["feature_set"])
        & (df["model_name"] == row["model_name"])
    )
    if not df.empty:
        df = df.loc[~mask].copy()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values(
        ["task", "experiment_group", "feature_set", "val_auprc", "test_auprc"],
        ascending=[True, True, True, False, False],
    ).reset_index(drop=True)
    save_results(df)


def to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=to_serializable), encoding="utf-8")


def split_file(task_name: str) -> Path:
    return SPLITS_DIR / f"{task_name}_split_indices.json"


def selection_json_file(task_name: str) -> Path:
    return ARTIFACTS_DIR / f"{task_name}_feature_selection.json"


def selection_state_file(task_name: str) -> Path:
    return ARTIFACTS_DIR / f"{task_name}_feature_selection.joblib"


def selection_csv_file(task_name: str) -> Path:
    return REPORTS_DIR / f"{task_name}_selected_features__{INPUT_CSV.stem}.csv"


def model_file(task_name: str, experiment_group: str, feature_set: str, model_name: str) -> Path:
    return MODELS_DIR / f"{task_name}__{experiment_group}__{feature_set}__{model_name}.joblib"


def artifact_file(task_name: str, experiment_group: str, feature_set: str, model_name: str) -> Path:
    return ARTIFACTS_DIR / f"{task_name}__{experiment_group}__{feature_set}__{model_name}.json"


def delong_file(task_name: str) -> Path:
    return REPORTS_DIR / f"{task_name}_delong.json"


def shap_file(task_name: str) -> Path:
    return REPORTS_DIR / f"{task_name}_shap.json"


def load_dataset() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Missing modeling dataset: {INPUT_CSV.name}. Run build_region_tornado_dataset.py first."
        )
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    required = set()
    for spec in FEATURE_SPECS.values():
        required.update(spec.all_features)
    required.update(TASKS.values())
    missing = sorted(col for col in required if col not in df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def build_preprocessor(spec: FeatureSpec, categories: list[list[object]] | None = None) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    encoder_kwargs: dict[str, object] = {"handle_unknown": "ignore", "sparse_output": False}
    if categories is not None:
        encoder_kwargs["categories"] = categories
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(**encoder_kwargs)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, spec.numeric_features),
            ("cat", categorical_pipe, spec.categorical_features),
        ],
        remainder="drop",
    )


def split_dataset(df: pd.DataFrame, target_col: str) -> dict[str, pd.Index]:
    y = df[target_col]
    idx = df.index.to_numpy()
    train_val_idx, test_idx = train_test_split(
        idx,
        test_size=TEST_SHARE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    train_val_y = y.loc[train_val_idx]
    val_relative_share = VAL_SHARE / (TRAIN_SHARE + VAL_SHARE)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_relative_share,
        stratify=train_val_y,
        random_state=RANDOM_STATE,
    )
    return {"train": pd.Index(train_idx), "val": pd.Index(val_idx), "test": pd.Index(test_idx)}


def ensure_split_indices(
    df: pd.DataFrame, task_name: str, target_col: str, force: bool = False
) -> dict[str, pd.Index]:
    path = split_file(task_name)
    if path.exists() and not force:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {key: pd.Index(payload[key]) for key in ["train", "val", "test"]}

    split_idx = split_dataset(df, target_col)
    save_json(
        path,
        {
            "task": task_name,
            "target_col": target_col,
            "train_share": TRAIN_SHARE,
            "val_share": VAL_SHARE,
            "test_share": TEST_SHARE,
            "random_state": RANDOM_STATE,
            "train": [int(x) for x in split_idx["train"].tolist()],
            "val": [int(x) for x in split_idx["val"].tolist()],
            "test": [int(x) for x in split_idx["test"].tolist()],
        },
    )
    return split_idx


def build_cv(cv_splits: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)


def build_baseline_search(spec: FeatureSpec, cv_splits: int) -> GridSearchCV:
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(spec)),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return GridSearchCV(
        estimator=pipeline,
        param_grid={"classifier__C": BASELINE_C_GRID},
        cv=build_cv(cv_splits),
        scoring="average_precision",
        refit=True,
        n_jobs=1,
        error_score="raise",
    )


def fit_feature_selector(
    df: pd.DataFrame,
    task_name: str,
    target_col: str,
    split_idx: dict[str, pd.Index],
    cv_splits: int,
    force: bool,
) -> dict[str, object]:
    joblib_path = selection_state_file(task_name)
    if joblib_path.exists() and not force:
        return joblib.load(joblib_path)

    spec = FEATURE_SPECS["county_enriched"]
    x_train = df.loc[split_idx["train"], spec.all_features]
    y_train = df.loc[split_idx["train"], target_col]

    log(f"Running explicit L1 feature selection for {task_name} on county-enriched features...")
    preprocessor = build_preprocessor(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preprocessor.fit(x_train)
    x_train_transformed = preprocessor.transform(x_train)
    transformed_names = preprocessor.get_feature_names_out().tolist()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_selector = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            C=SELECTOR_C,
            max_iter=2000,
            tol=1e-3,
            random_state=RANDOM_STATE,
        ).fit(x_train_transformed, y_train)
    coefficients = np.asarray(best_selector.coef_).ravel()
    selected_mask = np.abs(coefficients) > NONZERO_EPS
    if not selected_mask.any():
        selected_mask[int(np.argmax(np.abs(coefficients)))] = True

    selected_indices = np.flatnonzero(selected_mask).tolist()
    if len(selected_indices) > SELECTOR_MAX_FEATURES:
        ranked = sorted(selected_indices, key=lambda idx: abs(coefficients[idx]), reverse=True)
        selected_indices = ranked[:SELECTOR_MAX_FEATURES]
    selected_names = [transformed_names[idx] for idx in selected_indices]
    selected_coefs = [float(coefficients[idx]) for idx in selected_indices]

    onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    category_levels = [list(levels) for levels in onehot.categories_]

    state = {
        "task": task_name,
        "target_col": target_col,
        "feature_set": "county_enriched",
        "cv_splits": cv_splits,
        "selector_model": "l1_logistic_regression",
        "selector_c": float(SELECTOR_C),
        "selector_max_features": int(SELECTOR_MAX_FEATURES),
        "transformed_feature_count": len(transformed_names),
        "selected_feature_count": len(selected_names),
        "selected_indices": selected_indices,
        "selected_features": selected_names,
        "selected_coefficients": selected_coefs,
        "categorical_levels": category_levels,
        "notes": [
            "Feature selection is run once on the training split only, before downstream model comparison.",
            "L1 logistic coefficients are ranked by absolute value and capped to the top features requested for this run.",
            "No cross-validation is used for feature selection in this simplified pass.",
            "The training-derived category levels are frozen for downstream CV so the selected columns stay aligned.",
        ],
    }

    joblib.dump(state, joblib_path)
    save_json(selection_json_file(task_name), state)
    pd.DataFrame(
        {
            "selected_feature": selected_names,
            "coefficient": selected_coefs,
            "abs_coefficient": np.abs(selected_coefs),
        }
    ).sort_values("abs_coefficient", ascending=False).to_csv(selection_csv_file(task_name), index=False)

    log(
        f"Selected {state['selected_feature_count']} of {state['transformed_feature_count']} transformed features "
        f"for {task_name}."
    )
    return state


def load_selection_state(task_name: str) -> dict[str, object]:
    path = selection_state_file(task_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature-selection artifact for {task_name}. Run --stage feature_selection first."
        )
    return joblib.load(path)


def build_selected_pipeline(selection_state: dict[str, object], classifier) -> Pipeline:
    spec = FEATURE_SPECS["county_enriched"]
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(spec, categories=selection_state["categorical_levels"])),
            (
                "selector",
                FixedFeatureSelector(
                    selected_indices=selection_state["selected_indices"],
                    selected_names=selection_state["selected_features"],
                ),
            ),
            ("classifier", classifier),
        ]
    )


def build_model_search(
    model_name: str,
    selection_state: dict[str, object],
    y_train: pd.Series,
    cv_splits: int,
) -> tuple[GridSearchCV | RandomizedSearchCV, dict[str, object], str]:
    common = {
        "cv": build_cv(cv_splits),
        "scoring": "average_precision",
        "refit": True,
        "n_jobs": 1,
        "error_score": "raise",
    }
    pos_weight = float((len(y_train) - y_train.sum()) / max(float(y_train.sum()), 1.0))

    if model_name == "logistic_regression":
        pipeline = build_selected_pipeline(
            selection_state,
            LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                class_weight="balanced",
                max_iter=5000,
                random_state=RANDOM_STATE,
            ),
        )
        return (
            GridSearchCV(
                estimator=pipeline,
                param_grid={"classifier__C": [0.05, 0.1, 0.5, 1.0, 5.0]},
                **common,
            ),
            {},
            "class_weight=balanced",
        )

    if model_name == "random_forest":
        pipeline = build_selected_pipeline(
            selection_state,
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=TREE_WORKERS,
                class_weight="balanced_subsample",
            ),
        )
        return (
            RandomizedSearchCV(
                estimator=pipeline,
                param_distributions={
                    "classifier__n_estimators": [300, 500],
                    "classifier__max_depth": [None, 12, 20],
                    "classifier__min_samples_leaf": [1, 3, 5],
                    "classifier__max_features": ["sqrt", 0.5],
                },
                n_iter=RANDOM_SEARCH_ITERS,
                random_state=RANDOM_STATE,
                **common,
            ),
            {},
            "class_weight=balanced_subsample",
        )

    if model_name == "xgboost":
        pipeline = build_selected_pipeline(
            selection_state,
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="aucpr",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=TREE_WORKERS,
                scale_pos_weight=pos_weight,
                verbosity=0,
            ),
        )
        return (
            RandomizedSearchCV(
                estimator=pipeline,
                param_distributions={
                    "classifier__n_estimators": [200, 350],
                    "classifier__max_depth": [3, 5, 7],
                    "classifier__learning_rate": [0.03, 0.08, 0.15],
                    "classifier__min_child_weight": [1, 3, 5],
                    "classifier__subsample": [0.8, 1.0],
                    "classifier__colsample_bytree": [0.7, 1.0],
                },
                n_iter=RANDOM_SEARCH_ITERS,
                random_state=RANDOM_STATE,
                **common,
            ),
            {},
            f"scale_pos_weight={pos_weight:.4f}",
        )

    if model_name == "svm":
        pipeline = build_selected_pipeline(
            selection_state,
            LinearSVC(
                class_weight="balanced",
                dual="auto",
                max_iter=10000,
                random_state=RANDOM_STATE,
            ),
        )
        return (
            GridSearchCV(
                estimator=pipeline,
                param_grid={"classifier__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]},
                **common,
            ),
            {},
            "class_weight=balanced; linear_svm_for_tractability",
        )

    if model_name == "neural_network":
        pipeline = build_selected_pipeline(
            selection_state,
            MLPClassifier(
                early_stopping=True,
                max_iter=300,
                random_state=RANDOM_STATE,
            ),
        )
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        return (
            RandomizedSearchCV(
                estimator=pipeline,
                param_distributions={
                    "classifier__hidden_layer_sizes": [(64,), (128,), (64, 32)],
                    "classifier__alpha": [0.0001, 0.001, 0.01],
                    "classifier__learning_rate_init": [0.001, 0.003],
                },
                n_iter=RANDOM_SEARCH_ITERS,
                random_state=RANDOM_STATE,
                **common,
            ),
            {"classifier__sample_weight": sample_weight},
            "balanced_sample_weight",
        )

    if model_name == "knn":
        pipeline = build_selected_pipeline(
            selection_state,
            KNeighborsClassifier(weights="distance"),
        )
        return (
            GridSearchCV(
                estimator=pipeline,
                param_grid={
                    "classifier__n_neighbors": [11, 21, 31, 41],
                    "classifier__p": [1, 2],
                },
                **common,
            ),
            {},
            "no_native_class_weight_support_in_sklearn_knn; distance_weighting_only",
        )

    raise ValueError(f"Unsupported model: {model_name}")


def get_scores(model, x: pd.DataFrame | np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    raise TypeError("Model must expose predict_proba or decision_function.")


def find_best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5
    f1_values = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.nanargmax(f1_values))
    return float(thresholds[best_idx])


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


def get_feature_names(model: Pipeline) -> list[str]:
    preprocess = model.named_steps["preprocess"]
    names = preprocess.get_feature_names_out().tolist()
    selector = model.named_steps.get("selector")
    if selector is None:
        return names
    support = selector.get_support()
    return [name for name, keep in zip(names, support) if keep]


def transform_selected(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    transformed = model.named_steps["preprocess"].transform(x)
    selector = model.named_steps.get("selector")
    if selector is None:
        return transformed
    return selector.transform(transformed)


def compute_midrank(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    order = np.argsort(x)
    sorted_x = x[order]
    midranks = np.zeros(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        midrank = 0.5 * (i + j - 1) + 1
        midranks[i:j] = midrank
        i = j
    out = np.empty(len(x), dtype=float)
    out[order] = midranks
    return out


def fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for r in range(k):
        tx[r] = compute_midrank(positive_examples[r])
        ty[r] = compute_midrank(negative_examples[r])
        tz[r] = compute_midrank(predictions_sorted_transposed[r])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_roc_test(y_true: np.ndarray, pred_one: np.ndarray, pred_two: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true)
    label_1_count = int(y_true.sum())
    preds = np.vstack([pred_one, pred_two])[:, order]
    aucs, covariance = fast_delong(preds, label_1_count)
    diff = np.array([[1, -1]], dtype=float)
    covariance = np.atleast_2d(covariance)
    variance = float((diff @ covariance @ diff.T).item())
    if variance <= 0:
        return {"auc_1": float(aucs[0]), "auc_2": float(aucs[1]), "z": math.inf, "p_value": 0.0}
    z_score = abs(float(aucs[0] - aucs[1])) / math.sqrt(variance)
    p_value = 2 * (1 - norm.cdf(z_score))
    return {"auc_1": float(aucs[0]), "auc_2": float(aucs[1]), "z": float(z_score), "p_value": float(p_value)}


def existing_result(task_name: str, experiment_group: str, feature_set: str, model_name: str) -> bool:
    df = load_results()
    if df.empty:
        return False
    mask = (
        (df["task"] == task_name)
        & (df["experiment_group"] == experiment_group)
        & (df["feature_set"] == feature_set)
        & (df["model_name"] == model_name)
    )
    return bool(mask.any())


def should_skip(task_name: str, experiment_group: str, feature_set: str, model_name: str, force: bool) -> bool:
    if force:
        return False
    return model_file(task_name, experiment_group, feature_set, model_name).exists() and existing_result(
        task_name, experiment_group, feature_set, model_name
    )


def fit_and_record(
    *,
    df: pd.DataFrame,
    task_name: str,
    target_col: str,
    split_idx: dict[str, pd.Index],
    spec: FeatureSpec,
    experiment_group: str,
    feature_set: str,
    model_name: str,
    search: GridSearchCV | RandomizedSearchCV,
    fit_kwargs: dict[str, object],
    cv_splits: int,
    imbalance_strategy: str,
) -> dict[str, object]:
    x_train = df.loc[split_idx["train"], spec.all_features]
    x_val = df.loc[split_idx["val"], spec.all_features]
    x_test = df.loc[split_idx["test"], spec.all_features]
    y_train = df.loc[split_idx["train"], target_col]
    y_val = df.loc[split_idx["val"], target_col]
    y_test = df.loc[split_idx["test"], target_col]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(x_train, y_train, **fit_kwargs)

    fitted = search.best_estimator_
    val_scores = get_scores(fitted, x_val)
    threshold = find_best_f1_threshold(y_val.to_numpy(), val_scores)
    test_scores = get_scores(fitted, x_test)
    val_metrics = evaluate_split(y_val.to_numpy(), val_scores, threshold)
    test_metrics = evaluate_split(y_test.to_numpy(), test_scores, threshold)
    selected_names = get_feature_names(fitted)

    row = {
        "task": task_name,
        "experiment_group": experiment_group,
        "feature_set": feature_set,
        "model_name": model_name,
        "cv_splits": cv_splits,
        "imbalance_strategy": imbalance_strategy,
        "cv_best_auprc": float(search.best_score_),
        "validation_threshold": float(threshold),
        "selected_feature_count": len(selected_names),
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
    }
    upsert_result(row)

    joblib.dump(fitted, model_file(task_name, experiment_group, feature_set, model_name))
    save_json(
        artifact_file(task_name, experiment_group, feature_set, model_name),
        {
            "task": task_name,
            "target_col": target_col,
            "experiment_group": experiment_group,
            "feature_set": feature_set,
            "model_name": model_name,
            "cv_splits": cv_splits,
            "imbalance_strategy": imbalance_strategy,
            "best_params": search.best_params_,
            "threshold": threshold,
            "selected_feature_count": len(selected_names),
            "selected_features": selected_names,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
    )
    return row


def run_baselines_for_task(
    df: pd.DataFrame,
    task_name: str,
    target_col: str,
    split_idx: dict[str, pd.Index],
    cv_splits: int,
    force: bool,
) -> None:
    for feature_set in ["tornado_only", "county_enriched"]:
        experiment_group = "baseline_logistic"
        model_name = "logistic_regression"
        if should_skip(task_name, experiment_group, feature_set, model_name, force):
            log(f"Skipping existing baseline for {task_name} / {feature_set}.")
            continue
        log(f"Running baseline logistic for {task_name} / {feature_set}...")
        spec = FEATURE_SPECS[feature_set]
        search = build_baseline_search(spec, cv_splits)
        row = fit_and_record(
            df=df,
            task_name=task_name,
            target_col=target_col,
            split_idx=split_idx,
            spec=spec,
            experiment_group=experiment_group,
            feature_set=feature_set,
            model_name=model_name,
            search=search,
            fit_kwargs={},
            cv_splits=cv_splits,
            imbalance_strategy="class_weight=balanced",
        )
        log(
            f"Saved baseline {task_name} / {feature_set}: "
            f"test AUROC={row['test_auroc']:.4f}, test AUPRC={row['test_auprc']:.4f}, test F1={row['test_f1']:.4f}"
        )


def run_model_for_task(
    df: pd.DataFrame,
    task_name: str,
    target_col: str,
    split_idx: dict[str, pd.Index],
    model_name: str,
    cv_splits: int,
    force: bool,
) -> None:
    experiment_group = "model_suite"
    feature_set = "county_selected"
    if should_skip(task_name, experiment_group, feature_set, model_name, force):
        log(f"Skipping existing suite model for {task_name} / {model_name}.")
        return

    selection_state = load_selection_state(task_name)
    y_train = df.loc[split_idx["train"], target_col]
    search, fit_kwargs, imbalance_strategy = build_model_search(model_name, selection_state, y_train, cv_splits)
    log(f"Running {model_name} for {task_name} on the selected county-enriched feature set...")
    row = fit_and_record(
        df=df,
        task_name=task_name,
        target_col=target_col,
        split_idx=split_idx,
        spec=FEATURE_SPECS["county_enriched"],
        experiment_group=experiment_group,
        feature_set=feature_set,
        model_name=model_name,
        search=search,
        fit_kwargs=fit_kwargs,
        cv_splits=cv_splits,
        imbalance_strategy=imbalance_strategy,
    )
    log(
        f"Saved {task_name} / {model_name}: "
        f"test AUROC={row['test_auroc']:.4f}, test AUPRC={row['test_auprc']:.4f}, test F1={row['test_f1']:.4f}"
    )


def run_shap_summary(
    fitted_model: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, object]:
    feature_names = get_feature_names(fitted_model)
    x_train_selected = transform_selected(fitted_model, x_train)
    x_test_selected = transform_selected(fitted_model, x_test)
    classifier = fitted_model.named_steps["classifier"]

    background_n = min(SHAP_BACKGROUND_N, len(x_train_selected))
    eval_n = min(SHAP_EVAL_N, len(x_test_selected))
    rng = np.random.default_rng(RANDOM_STATE)
    background_idx = rng.choice(len(x_train_selected), size=background_n, replace=False)
    eval_idx = rng.choice(len(x_test_selected), size=eval_n, replace=False)

    background = x_train_selected[background_idx]
    evaluation = x_test_selected[eval_idx]
    evaluation_y = y_test.iloc[eval_idx].reset_index(drop=True)

    if hasattr(classifier, "predict_proba"):
        predict_fn = lambda arr: classifier.predict_proba(arr)[:, 1]
    else:
        predict_fn = classifier.decision_function

    explainer = shap.Explainer(predict_fn, background, feature_names=feature_names)
    shap_values = explainer(evaluation)
    values = np.asarray(shap_values.values)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(-mean_abs)
    top_global = [
        {"feature": feature_names[idx], "mean_abs_shap": float(mean_abs[idx])}
        for idx in order[:15]
    ]

    scores = get_scores(classifier, evaluation)
    case_positions = [
        ("highest_risk", int(np.argmax(scores))),
        ("lowest_risk", int(np.argmin(scores))),
    ]
    local_cases = []
    for case_label, row_idx in case_positions:
        row_values = values[row_idx]
        row_order = np.argsort(-np.abs(row_values))[:10]
        contributions = [
            {
                "feature": feature_names[col_idx],
                "shap_value": float(row_values[col_idx]),
                "feature_value": float(evaluation[row_idx, col_idx]),
            }
            for col_idx in row_order
        ]
        local_cases.append(
            {
                "case_label": case_label,
                "predicted_score": float(scores[row_idx]),
                "true_label": int(evaluation_y.iloc[row_idx]),
                "top_contributions": contributions,
            }
        )

    return {
        "selected_feature_count": len(feature_names),
        "top_global_features": top_global,
        "local_cases": local_cases,
    }


def run_post_analysis_for_task(
    df: pd.DataFrame,
    task_name: str,
    target_col: str,
    split_idx: dict[str, pd.Index],
) -> None:
    results = load_results()
    task_results = results.loc[results["task"] == task_name].copy()
    if task_results.empty:
        raise ValueError(f"No saved results found for {task_name}. Run baselines/models first.")

    task_results = task_results.sort_values(
        ["val_auprc", "val_f1", "test_auprc", "test_auroc"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    best_row = task_results.iloc[0]

    delong_payload: dict[str, object]
    if len(task_results) >= 2:
        first = task_results.iloc[0]
        second = task_results.iloc[1]
        y_test = df.loc[split_idx["test"], target_col].to_numpy()
        first_model = joblib.load(
            model_file(first["task"], first["experiment_group"], first["feature_set"], first["model_name"])
        )
        second_model = joblib.load(
            model_file(second["task"], second["experiment_group"], second["feature_set"], second["model_name"])
        )

        first_spec = FEATURE_SPECS["county_enriched"] if first["feature_set"] != "tornado_only" else FEATURE_SPECS["tornado_only"]
        second_spec = FEATURE_SPECS["county_enriched"] if second["feature_set"] != "tornado_only" else FEATURE_SPECS["tornado_only"]
        x_test_first = df.loc[split_idx["test"], first_spec.all_features]
        x_test_second = df.loc[split_idx["test"], second_spec.all_features]
        first_scores = get_scores(first_model, x_test_first)
        second_scores = get_scores(second_model, x_test_second)
        delong_payload = {
            "task": task_name,
            "model_1": {
                "experiment_group": first["experiment_group"],
                "feature_set": first["feature_set"],
                "model_name": first["model_name"],
            },
            "model_2": {
                "experiment_group": second["experiment_group"],
                "feature_set": second["feature_set"],
                "model_name": second["model_name"],
            },
            "test_set_delong": delong_roc_test(y_test, first_scores, second_scores),
        }
    else:
        delong_payload = {
            "task": task_name,
            "note": "Only one completed model was available, so DeLong comparison was skipped.",
        }
    save_json(delong_file(task_name), delong_payload)

    best_model = joblib.load(
        model_file(best_row["task"], best_row["experiment_group"], best_row["feature_set"], best_row["model_name"])
    )
    best_spec = FEATURE_SPECS["county_enriched"] if best_row["feature_set"] != "tornado_only" else FEATURE_SPECS["tornado_only"]
    x_train = df.loc[split_idx["train"], best_spec.all_features]
    x_test = df.loc[split_idx["test"], best_spec.all_features]
    y_test = df.loc[split_idx["test"], target_col]

    shap_payload = {
        "task": task_name,
        "selected_by": "validation_auprc_then_validation_f1",
        "best_model": {
            "experiment_group": best_row["experiment_group"],
            "feature_set": best_row["feature_set"],
            "model_name": best_row["model_name"],
        },
        "shap_summary": run_shap_summary(best_model, x_train, x_test, y_test),
    }
    save_json(shap_file(task_name), shap_payload)


def write_methodology_decisions(cv_splits: int) -> None:
    payload = {
        "input_dataset": str(INPUT_CSV),
        "random_state": RANDOM_STATE,
        "splits": {
            "train_share": TRAIN_SHARE,
            "validation_share": VAL_SHARE,
            "test_share": TEST_SHARE,
        },
        "cv_splits": cv_splits,
        "pdf_alignment": [
            "Cleaning and format handling comes from the existing final_tornado_dataset.csv built from the PDF-described preprocessing.",
            "The modeling dataset replaces state codes with broader NOAA-style regions before any modeling.",
            "The modeling candidate set keeps only PDF-aligned storm, season/location, and county vulnerability/exposure variables.",
            "Baselines are logistic regression on tornado-only and county-enriched feature sets.",
            "Feature selection is a separate explicit stage using L1-regularized logistic regression on the training split only.",
            "Downstream model comparison runs on the selected county-enriched feature set.",
            "Thresholds are chosen on the validation split by maximizing validation F1 and then applied to the test split.",
            "Final post-processing includes DeLong AUROC comparison and SHAP for the best validation model.",
        ],
        "pragmatic_decisions": [
            "Cross-validation defaults to 3 folds because the user explicitly requested a faster same-day completion path.",
            "Feature selection uses a single fixed L1-logistic penalty strength (C=0.05) instead of a selector CV sweep, per user request to keep this stage simple.",
            "When more than 20 coefficients survive the L1 fit, the selector keeps the top 20 by absolute coefficient magnitude.",
            "States and territories are replaced with broader regions so the selector cannot spend feature budget on state-level dummies.",
            "Raw coordinates, raw year/month/day fields, segmentation diagnostics, overlay metadata flags, and duplicate magnitude encoding are excluded before feature selection.",
            "SVM is implemented as LinearSVC for tractability at 67k rows.",
            "KNN is kept in the comparison, but sklearn KNN has no native class-weight support, so no resampling is added.",
            "Neural-network imbalance handling uses balanced sample weights, which sklearn MLP supports directly.",
        ],
        "imbalance_handling": {
            "baseline_logistic": "class_weight=balanced",
            "logistic_regression": "class_weight=balanced",
            "random_forest": "class_weight=balanced_subsample",
            "xgboost": "scale_pos_weight=neg_pos_ratio",
            "svm": "class_weight=balanced",
            "neural_network": "balanced sample weights passed to fit",
            "knn": "no native class-weight support; distance weighting only",
        },
    }
    save_json(METHODOLOGY_JSON, payload)


def make_markdown_summary() -> str:
    results = load_results()
    lines = [
        "# Tornado ML Project Summary",
        "",
        f"- Input dataset: `{INPUT_CSV.name}`",
        f"- Results table: `{RESULTS_CSV.name}`",
        f"- Methodology log: `{METHODOLOGY_JSON.name}`",
        "",
        "## Current Results",
    ]
    if results.empty:
        lines.append("")
        lines.append("No experiment results have been saved yet.")
        return "\n".join(lines)

    for task_name in TASKS:
        task_results = results.loc[results["task"] == task_name].copy()
        lines.append("")
        lines.append(f"### {task_name.title()}")
        if task_results.empty:
            lines.append("No completed runs yet.")
            continue
        task_results = task_results.sort_values(
            ["val_auprc", "test_auprc", "test_auroc"], ascending=[False, False, False]
        )
        display_cols = [
            "experiment_group",
            "feature_set",
            "model_name",
            "selected_feature_count",
            "val_auprc",
            "test_auprc",
            "test_auroc",
            "test_f1",
        ]
        lines.extend(task_results[display_cols].to_string(index=False).splitlines())
    return "\n".join(lines)


def write_summary() -> None:
    SUMMARY_MD.write_text(make_markdown_summary(), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the tornado injury/fatality modeling pipeline described in ML_Project.pdf."
    )
    parser.add_argument(
        "--task",
        choices=["all", *TASKS.keys()],
        default="all",
        help="Which prediction task to run.",
    )
    parser.add_argument(
        "--stage",
        choices=["baselines", "feature_selection", "model", "post", "full"],
        required=True,
        help="Which pipeline stage to run.",
    )
    parser.add_argument(
        "--model",
        choices=["all", *MODEL_ORDER],
        default="all",
        help="Model to run for --stage model or --stage full.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=DEFAULT_CV_SPLITS,
        help="Cross-validation folds for tuning. Defaults to 3 for speed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute outputs even if saved artifacts already exist.",
    )
    return parser.parse_args()


def task_list(task_arg: str) -> list[str]:
    return list(TASKS.keys()) if task_arg == "all" else [task_arg]


def model_list(model_arg: str) -> list[str]:
    return MODEL_ORDER if model_arg == "all" else [model_arg]


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    write_methodology_decisions(args.cv_splits)

    df = load_dataset()
    tasks = task_list(args.task)
    models = model_list(args.model)

    for task_name in tasks:
        target_col = TASKS[task_name]
        split_idx = ensure_split_indices(df, task_name, target_col, force=args.force)

        if args.stage in {"baselines", "full"}:
            run_baselines_for_task(df, task_name, target_col, split_idx, args.cv_splits, args.force)
            write_summary()

        if args.stage in {"feature_selection", "full"}:
            fit_feature_selector(df, task_name, target_col, split_idx, args.cv_splits, args.force)

        if args.stage in {"model", "full"}:
            for model_name in models:
                run_model_for_task(df, task_name, target_col, split_idx, model_name, args.cv_splits, args.force)
                write_summary()

        if args.stage in {"post", "full"}:
            run_post_analysis_for_task(df, task_name, target_col, split_idx)
            write_summary()

    log("Finished requested stage(s).")


if __name__ == "__main__":
    main()
