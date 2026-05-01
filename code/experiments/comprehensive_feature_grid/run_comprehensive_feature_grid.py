from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_tornado_ml_project as project


OUTPUT_DIR = ROOT / "outputs"
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_CSV = OUTPUT_DIR / "comprehensive_feature_grid_results.csv"
SUMMARY_JSON = OUTPUT_DIR / "comprehensive_feature_grid_summary.json"
LATEX_SNIPPET = OUTPUT_DIR / "appendix_comprehensive_feature_grid_tables.tex"
EXISTING_RESULTS_CSV = PROJECT_ROOT / "final out" / "tables" / "full_model_results_detailed.csv"


ALL_FEATURE_SETS = ["tornado_only", "county_enriched", "county_selected"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def ensure_dirs() -> None:
    for path in [OUTPUT_DIR, ARTIFACTS_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def artifact_path(task_name: str, feature_set: str, model_name: str) -> Path:
    return ARTIFACTS_DIR / f"{task_name}__{feature_set}__{model_name}.json"


def model_path(task_name: str, feature_set: str, model_name: str) -> Path:
    return MODELS_DIR / f"{task_name}__{feature_set}__{model_name}.joblib"


def existing_rows() -> dict[tuple[str, str, str], dict[str, object]]:
    df = pd.read_csv(EXISTING_RESULTS_CSV)
    rows: dict[tuple[str, str, str], dict[str, object]] = {}
    for _, row in df.iterrows():
        key = (str(row["task"]), str(row["feature_set"]), str(row["model_name"]))
        rows[key] = row.to_dict()
    return rows


def saved_output_rows() -> dict[tuple[str, str, str], dict[str, object]]:
    rows: dict[tuple[str, str, str], dict[str, object]] = {}
    if not ARTIFACTS_DIR.exists():
        return rows
    for path in sorted(ARTIFACTS_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        key = (str(payload["task"]), str(payload["feature_set"]), str(payload["model_name"]))
        rows[key] = payload
    return rows


def build_nonselected_search(
    feature_set: str,
    model_name: str,
    spec: project.FeatureSpec,
    y_train: pd.Series,
    cv_splits: int,
) -> tuple[GridSearchCV | RandomizedSearchCV, dict[str, object], str]:
    common = {
        "cv": project.build_cv(cv_splits),
        "scoring": "average_precision",
        "refit": True,
        "n_jobs": 1,
        "error_score": "raise",
    }
    pos_weight = float((len(y_train) - y_train.sum()) / max(float(y_train.sum()), 1.0))
    preprocess = project.build_preprocessor(spec)

    if model_name == "logistic_regression":
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocess),
                (
                    "classifier",
                    LogisticRegression(
                        penalty="l2",
                        solver="lbfgs",
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=project.RANDOM_STATE,
                    ),
                ),
            ]
        )
        grid = project.BASELINE_C_GRID if feature_set != "county_selected" else [0.05, 0.1, 0.5, 1.0, 5.0]
        return (
            GridSearchCV(estimator=pipeline, param_grid={"classifier__C": grid}, **common),
            {},
            "class_weight=balanced",
        )

    if model_name == "random_forest":
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocess),
                (
                    "classifier",
                    RandomForestClassifier(
                        random_state=project.RANDOM_STATE,
                        n_jobs=1,
                        class_weight="balanced_subsample",
                    ),
                ),
            ]
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
                n_iter=project.RANDOM_SEARCH_ITERS,
                random_state=project.RANDOM_STATE,
                **common,
            ),
            {},
            "class_weight=balanced_subsample",
        )

    if model_name == "xgboost":
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocess),
                (
                    "classifier",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="aucpr",
                        tree_method="hist",
                        random_state=project.RANDOM_STATE,
                        n_jobs=1,
                        scale_pos_weight=pos_weight,
                        verbosity=0,
                    ),
                ),
            ]
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
                n_iter=project.RANDOM_SEARCH_ITERS,
                random_state=project.RANDOM_STATE,
                **common,
            ),
            {},
            f"scale_pos_weight={pos_weight:.4f}",
        )

    if model_name == "svm":
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocess),
                (
                    "classifier",
                    LinearSVC(
                        class_weight="balanced",
                        dual="auto",
                        max_iter=10000,
                        random_state=project.RANDOM_STATE,
                    ),
                ),
            ]
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
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocess),
                (
                    "classifier",
                    MLPClassifier(
                        early_stopping=True,
                        max_iter=300,
                        random_state=project.RANDOM_STATE,
                    ),
                ),
            ]
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
                n_iter=project.RANDOM_SEARCH_ITERS,
                random_state=project.RANDOM_STATE,
                **common,
            ),
            {"classifier__sample_weight": sample_weight},
            "balanced_sample_weight",
        )

    if model_name == "knn":
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("classifier", KNeighborsClassifier(weights="distance")),
            ]
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


def run_new_combo(
    df: pd.DataFrame,
    task_name: str,
    target_col: str,
    feature_set: str,
    model_name: str,
    cv_splits: int,
) -> dict[str, object]:
    split_idx = project.ensure_split_indices(df, task_name, target_col, force=False)
    y_train = df.loc[split_idx["train"], target_col]

    if feature_set == "county_selected":
        selection_state = project.load_selection_state(task_name)
        search, fit_kwargs, imbalance_strategy = project.build_model_search(
            model_name, selection_state, y_train, cv_splits
        )
        spec = project.FEATURE_SPECS["county_enriched"]
        selected_descriptor = "county_selected"
    else:
        spec = project.FEATURE_SPECS[feature_set]
        search, fit_kwargs, imbalance_strategy = build_nonselected_search(
            feature_set, model_name, spec, y_train, cv_splits
        )
        selected_descriptor = feature_set

    log(f"Running {task_name} / {feature_set} / {model_name}...")
    x_train = df.loc[split_idx["train"], spec.all_features]
    x_val = df.loc[split_idx["val"], spec.all_features]
    x_test = df.loc[split_idx["test"], spec.all_features]
    y_val = df.loc[split_idx["val"], target_col]
    y_test = df.loc[split_idx["test"], target_col]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(x_train, y_train, **fit_kwargs)

    fitted = search.best_estimator_
    val_scores = project.get_scores(fitted, x_val)
    threshold = project.find_best_f1_threshold(y_val.to_numpy(), val_scores)
    test_scores = project.get_scores(fitted, x_test)
    val_metrics = project.evaluate_split(y_val.to_numpy(), val_scores, threshold)
    test_metrics = project.evaluate_split(y_test.to_numpy(), test_scores, threshold)
    selected_names = project.get_feature_names(fitted)

    artifact = {
        "task": task_name,
        "target_col": target_col,
        "feature_set": feature_set,
        "model_name": model_name,
        "selected_descriptor": selected_descriptor,
        "cv_splits": cv_splits,
        "imbalance_strategy": imbalance_strategy,
        "best_params": search.best_params_,
        "cv_best_auprc": float(search.best_score_),
        "validation_threshold": float(threshold),
        "selected_feature_count": len(selected_names),
        "selected_features": selected_names,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    artifact_path(task_name, feature_set, model_name).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    joblib.dump(fitted, model_path(task_name, feature_set, model_name))

    return {
        "task": task_name,
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
        "val_tp": val_metrics["tp"],
        "val_fp": val_metrics["fp"],
        "val_tn": val_metrics["tn"],
        "val_fn": val_metrics["fn"],
        "test_tp": test_metrics["tp"],
        "test_fp": test_metrics["fp"],
        "test_tn": test_metrics["tn"],
        "test_fn": test_metrics["fn"],
        "best_params_json": json.dumps(search.best_params_, sort_keys=True),
        "source": "new_comprehensive_run",
    }


def normalized_existing_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "task": str(row["task"]),
        "feature_set": str(row["feature_set"]),
        "model_name": str(row["model_name"]),
        "cv_splits": int(row["cv_splits"]),
        "imbalance_strategy": str(row["imbalance_strategy"]),
        "cv_best_auprc": float(row["cv_best_auprc"]),
        "validation_threshold": float(row["validation_threshold"]),
        "selected_feature_count": int(float(row["selected_feature_count"])),
        "val_precision": float(row["val_precision"]),
        "val_recall": float(row["val_recall"]),
        "val_f1": float(row["val_f1"]),
        "val_auroc": float(row["val_auroc"]),
        "val_auprc": float(row["val_auprc"]),
        "test_precision": float(row["test_precision"]),
        "test_recall": float(row["test_recall"]),
        "test_f1": float(row["test_f1"]),
        "test_auroc": float(row["test_auroc"]),
        "test_auprc": float(row["test_auprc"]),
        "val_tp": int(row["val_tp"]),
        "val_fp": int(row["val_fp"]),
        "val_tn": int(row["val_tn"]),
        "val_fn": int(row["val_fn"]),
        "test_tp": int(row["test_tp"]),
        "test_fp": int(row["test_fp"]),
        "test_tn": int(row["test_tn"]),
        "test_fn": int(row["test_fn"]),
        "best_params_json": str(row["best_params_json"]),
        "source": "existing_final_run",
    }


def normalized_saved_output_row(payload: dict[str, object]) -> dict[str, object]:
    return {
        "task": str(payload["task"]),
        "feature_set": str(payload["feature_set"]),
        "model_name": str(payload["model_name"]),
        "cv_splits": int(payload["cv_splits"]),
        "imbalance_strategy": str(payload["imbalance_strategy"]),
        "cv_best_auprc": float(payload["cv_best_auprc"]),
        "validation_threshold": float(payload["validation_threshold"]),
        "selected_feature_count": int(payload["selected_feature_count"]),
        "val_precision": float(payload["val_metrics"]["precision"]),
        "val_recall": float(payload["val_metrics"]["recall"]),
        "val_f1": float(payload["val_metrics"]["f1"]),
        "val_auroc": float(payload["val_metrics"]["auroc"]),
        "val_auprc": float(payload["val_metrics"]["auprc"]),
        "test_precision": float(payload["test_metrics"]["precision"]),
        "test_recall": float(payload["test_metrics"]["recall"]),
        "test_f1": float(payload["test_metrics"]["f1"]),
        "test_auroc": float(payload["test_metrics"]["auroc"]),
        "test_auprc": float(payload["test_metrics"]["auprc"]),
        "val_tp": int(payload["val_metrics"]["tp"]),
        "val_fp": int(payload["val_metrics"]["fp"]),
        "val_tn": int(payload["val_metrics"]["tn"]),
        "val_fn": int(payload["val_metrics"]["fn"]),
        "test_tp": int(payload["test_metrics"]["tp"]),
        "test_fp": int(payload["test_metrics"]["fp"]),
        "test_tn": int(payload["test_metrics"]["tn"]),
        "test_fn": int(payload["test_metrics"]["fn"]),
        "best_params_json": json.dumps(payload["best_params"], sort_keys=True),
        "source": "saved_partial_output",
    }


def model_label(model_name: str, feature_set: str) -> str:
    if model_name == "logistic_regression" and feature_set == "county_selected":
        return "Selected logistic"
    if model_name == "logistic_regression":
        return "Baseline logistic"
    mapping = {
        "random_forest": "Random forest",
        "xgboost": "XGBoost",
        "svm": "Linear SVM",
        "neural_network": "Neural network",
        "knn": "KNN",
    }
    return mapping[model_name]


def feature_label(feature_set: str) -> str:
    return feature_set.replace("_", " ").title()


def latex_task_table(df: pd.DataFrame, task_name: str, label: str, caption: str) -> str:
    task_df = df.loc[df["task"] == task_name].copy()
    feature_order = {"tornado_only": 0, "county_enriched": 1, "county_selected": 2}
    model_order = {name: idx for idx, name in enumerate(project.MODEL_ORDER)}
    task_df["feature_order"] = task_df["feature_set"].map(feature_order)
    task_df["model_order"] = task_df["model_name"].map(model_order)
    task_df = task_df.sort_values(["feature_order", "model_order"]).reset_index(drop=True)

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lllccccccccccccc}",
        "\\toprule",
        "Model & Feature set & CV AUPRC & Val thr. & Sel. feats & Val Prec. & Val Rec. & Val F1 & Val AUROC & Val AUPRC & Test Prec. & Test Rec. & Test F1 & Test AUROC & Test AUPRC \\\\",
        "\\midrule",
    ]
    for _, row in task_df.iterrows():
        vals = [
            model_label(str(row["model_name"]), str(row["feature_set"])),
            feature_label(str(row["feature_set"])),
            f'{float(row["cv_best_auprc"]):.3f}',
            f'{float(row["validation_threshold"]):.3f}',
            str(int(row["selected_feature_count"])),
            f'{float(row["val_precision"]):.3f}',
            f'{float(row["val_recall"]):.3f}',
            f'{float(row["val_f1"]):.3f}',
            f'{float(row["val_auroc"]):.3f}',
            f'{float(row["val_auprc"]):.3f}',
            f'{float(row["test_precision"]):.3f}',
            f'{float(row["test_recall"]):.3f}',
            f'{float(row["test_f1"]):.3f}',
            f'{float(row["test_auroc"]):.3f}',
            f'{float(row["test_auprc"]):.3f}',
        ]
        lines.append(" & ".join(vals) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def write_latex_snippet(df: pd.DataFrame) -> None:
    content = "\n\n".join(
        [
            "Table~\\ref{tab:appendix_all_results_injury} and Table~\\ref{tab:appendix_all_results_fatality} report the full model-by-feature-set grid. Logistic regression was evaluated on all three feature sets, and the remaining model families were additionally evaluated on the tornado-only and county-enriched feature sets so that every model-feature-set combination is represented.",
            latex_task_table(
                df,
                "injury",
                "tab:appendix_all_results_injury",
                "Comprehensive injury-prediction performance across all model-feature-set combinations.",
            ),
            latex_task_table(
                df,
                "fatality",
                "tab:appendix_all_results_fatality",
                "Comprehensive fatality-prediction performance across all model-feature-set combinations.",
            ),
        ]
    )
    LATEX_SNIPPET.write_text(content, encoding="utf-8")


def persist_progress(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    results = pd.DataFrame(rows)
    feature_order = {"tornado_only": 0, "county_enriched": 1, "county_selected": 2}
    model_order = {name: idx for idx, name in enumerate(project.MODEL_ORDER)}
    results["feature_order"] = results["feature_set"].map(feature_order)
    results["model_order"] = results["model_name"].map(model_order)
    results = (
        results.sort_values(["task", "feature_order", "model_order"])
        .drop(columns=["feature_order", "model_order"])
        .reset_index(drop=True)
    )
    results.to_csv(RESULTS_CSV, index=False)
    write_latex_snippet(results)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "rows_written": int(len(results)),
                "tasks": list(project.TASKS.keys()),
                "feature_sets": ALL_FEATURE_SETS,
                "models": project.MODEL_ORDER,
                "results_csv": str(RESULTS_CSV),
                "latex_snippet": str(LATEX_SNIPPET),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    ensure_dirs()
    df = project.load_dataset()
    existing = existing_rows()
    saved = saved_output_rows()
    total = len(project.TASKS) * len(ALL_FEATURE_SETS) * len(project.MODEL_ORDER)

    rows: list[dict[str, object]] = []
    completed = 0
    for task_name, target_col in project.TASKS.items():
        for feature_set in ALL_FEATURE_SETS:
            for model_name in project.MODEL_ORDER:
                key = (task_name, feature_set, model_name)
                if key in existing:
                    rows.append(normalized_existing_row(existing[key]))
                    completed += 1
                    log(f"[{completed}/{total}] Reused existing final run for {task_name} / {feature_set} / {model_name}.")
                    persist_progress(rows)
                    continue
                if key in saved:
                    rows.append(normalized_saved_output_row(saved[key]))
                    completed += 1
                    log(f"[{completed}/{total}] Reused saved partial output for {task_name} / {feature_set} / {model_name}.")
                    persist_progress(rows)
                    continue
                rows.append(run_new_combo(df, task_name, target_col, feature_set, model_name, project.DEFAULT_CV_SPLITS))
                completed += 1
                log(f"[{completed}/{total}] Saved new run for {task_name} / {feature_set} / {model_name}.")
                persist_progress(rows)

    persist_progress(rows)
    log(f"Saved comprehensive results to {RESULTS_CSV}")
    log(f"Saved LaTeX snippet to {LATEX_SNIPPET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
