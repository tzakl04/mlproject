#!/usr/bin/env python3

from __future__ import annotations

import itertools
import json
import math
import shutil
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_tornado_ml_project as project

sys.modules["__main__"].FixedFeatureSelector = project.FixedFeatureSelector


ROOT = Path(__file__).resolve().parent
FINAL_DIR = ROOT / "final out"
TABLES_DIR = FINAL_DIR / "tables"
PLOTS_DIR = FINAL_DIR / "plots"
REPORT_PATH = FINAL_DIR / "master_report.md"

RESULTS_PATH = project.RESULTS_CSV


def ensure_dirs() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_results() -> pd.DataFrame:
    return pd.read_csv(RESULTS_PATH)


def load_split(task_name: str) -> dict[str, pd.Index]:
    payload = json.loads(project.split_file(task_name).read_text(encoding="utf-8"))
    return {key: pd.Index(payload[key]) for key in ["train", "val", "test"]}


def model_identifier(row: pd.Series) -> str:
    return f"{row['experiment_group']}|{row['feature_set']}|{row['model_name']}"


def spec_for_feature_set(feature_set: str) -> project.FeatureSpec:
    return project.FEATURE_SPECS["tornado_only"] if feature_set == "tornado_only" else project.FEATURE_SPECS["county_enriched"]


def load_artifact(row: pd.Series) -> dict:
    path = project.artifact_file(row["task"], row["experiment_group"], row["feature_set"], row["model_name"])
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(row: pd.Series):
    return joblib.load(project.model_file(row["task"], row["experiment_group"], row["feature_set"], row["model_name"]))


def dataset_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    split_tables: dict[str, pd.DataFrame] = {}
    for task_name, target_col in project.TASKS.items():
        split_idx = load_split(task_name)
        overall_pos = int(df[target_col].sum())
        overall_n = int(len(df))
        rows.append(
            {
                "task": task_name,
                "subset": "overall",
                "rows": overall_n,
                "positive_rows": overall_pos,
                "negative_rows": overall_n - overall_pos,
                "positive_rate": overall_pos / overall_n,
            }
        )
        split_rows = []
        for split_name in ["train", "val", "test"]:
            subset = df.loc[split_idx[split_name], target_col]
            pos = int(subset.sum())
            n = int(len(subset))
            split_rows.append(
                {
                    "task": task_name,
                    "subset": split_name,
                    "rows": n,
                    "positive_rows": pos,
                    "negative_rows": n - pos,
                    "positive_rate": pos / n,
                }
            )
            rows.append(split_rows[-1])
        split_tables[task_name] = pd.DataFrame(split_rows)
    return pd.DataFrame(rows), split_tables


def build_detailed_results(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in results_df.iterrows():
        artifact = load_artifact(row)
        out = row.to_dict()
        out["model_id"] = model_identifier(row)
        out["best_params_json"] = json.dumps(artifact["best_params"], sort_keys=True)
        out["val_tp"] = artifact["val_metrics"]["tp"]
        out["val_fp"] = artifact["val_metrics"]["fp"]
        out["val_tn"] = artifact["val_metrics"]["tn"]
        out["val_fn"] = artifact["val_metrics"]["fn"]
        out["test_tp"] = artifact["test_metrics"]["tp"]
        out["test_fp"] = artifact["test_metrics"]["fp"]
        out["test_tn"] = artifact["test_metrics"]["tn"]
        out["test_fn"] = artifact["test_metrics"]["fn"]
        out["selected_features_json"] = json.dumps(artifact["selected_features"])
        rows.append(out)
    detailed = pd.DataFrame(rows)
    sort_cols = ["task", "val_auprc", "val_f1", "test_auprc", "test_auroc"]
    return detailed.sort_values(sort_cols, ascending=[True, False, False, False, False]).reset_index(drop=True)


def compute_test_predictions(df: pd.DataFrame, results_df: pd.DataFrame) -> dict[str, dict[str, dict[str, object]]]:
    predictions: dict[str, dict[str, dict[str, object]]] = {task: {} for task in project.TASKS}
    for _, row in results_df.iterrows():
        task_name = row["task"]
        split_idx = load_split(task_name)
        target_col = project.TASKS[task_name]
        spec = spec_for_feature_set(row["feature_set"])
        x_test = df.loc[split_idx["test"], spec.all_features]
        y_test = df.loc[split_idx["test"], target_col].to_numpy()
        model = load_model(row)
        scores = project.get_scores(model, x_test)
        predictions[task_name][model_identifier(row)] = {
            "row": row.to_dict(),
            "y_test": y_test,
            "scores": scores,
        }
    return predictions


def compute_delong_tables(predictions: dict[str, dict[str, dict[str, object]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    same_model_rows = []
    for task_name, task_predictions in predictions.items():
        ids = sorted(task_predictions.keys())
        for left_id, right_id in itertools.combinations(ids, 2):
            left = task_predictions[left_id]
            right = task_predictions[right_id]
            left_row = left["row"]
            right_row = right["row"]
            stats = project.delong_roc_test(left["y_test"], left["scores"], right["scores"])
            record = {
                "task": task_name,
                "model_1_id": left_id,
                "model_1_experiment_group": left_row["experiment_group"],
                "model_1_feature_set": left_row["feature_set"],
                "model_1_name": left_row["model_name"],
                "model_2_id": right_id,
                "model_2_experiment_group": right_row["experiment_group"],
                "model_2_feature_set": right_row["feature_set"],
                "model_2_name": right_row["model_name"],
                "auc_1": stats["auc_1"],
                "auc_2": stats["auc_2"],
                "auc_diff": stats["auc_1"] - stats["auc_2"],
                "z": stats["z"],
                "p_value": stats["p_value"],
                "same_model_name": left_row["model_name"] == right_row["model_name"],
                "same_feature_set": left_row["feature_set"] == right_row["feature_set"],
                "same_experiment_group": left_row["experiment_group"] == right_row["experiment_group"],
            }
            all_rows.append(record)
            if record["same_model_name"] and left_id != right_id:
                same_model_rows.append(record)
    all_df = pd.DataFrame(all_rows).sort_values(["task", "p_value", "model_1_id", "model_2_id"]).reset_index(drop=True)
    same_df = pd.DataFrame(same_model_rows).sort_values(["task", "model_1_name", "p_value"]).reset_index(drop=True)
    return all_df, same_df


def significant_delong_plots(
    predictions: dict[str, dict[str, dict[str, object]]], delong_all: pd.DataFrame
) -> pd.DataFrame:
    records = []
    sig = delong_all.loc[delong_all["p_value"] < 0.05].reset_index(drop=True)
    for idx, row in sig.iterrows():
        task_name = row["task"]
        left = predictions[task_name][row["model_1_id"]]
        right = predictions[task_name][row["model_2_id"]]
        y_true = left["y_test"]
        left_scores = left["scores"]
        right_scores = right["scores"]

        fpr1, tpr1, _ = roc_curve(y_true, left_scores)
        fpr2, tpr2, _ = roc_curve(y_true, right_scores)

        plt.figure(figsize=(7, 6))
        plt.plot(fpr1, tpr1, label=f"{row['model_1_id']} (AUC={row['auc_1']:.4f})", linewidth=2)
        plt.plot(fpr2, tpr2, label=f"{row['model_2_id']} (AUC={row['auc_2']:.4f})", linewidth=2)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{task_name.title()} Significant DeLong Comparison\np={row['p_value']:.4g}")
        plt.legend(fontsize=8, loc="lower right")
        filename = f"{task_name}_delong_significant_{idx + 1:02d}.png"
        write_png(PLOTS_DIR / filename)
        records.append(
            {
                "task": task_name,
                "model_1_id": row["model_1_id"],
                "model_2_id": row["model_2_id"],
                "auc_1": row["auc_1"],
                "auc_2": row["auc_2"],
                "auc_diff": row["auc_diff"],
                "z": row["z"],
                "p_value": row["p_value"],
                "plot_file": filename,
            }
        )
    return pd.DataFrame(records)


def write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def shap_outputs_for_best_models(df: pd.DataFrame, results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    local_rows = []
    for task_name in project.TASKS:
        task_results = results_df.loc[results_df["task"] == task_name].copy()
        task_results = task_results.sort_values(
            ["val_auprc", "val_f1", "test_auprc", "test_auroc"], ascending=[False, False, False, False]
        )
        best_row = task_results.iloc[0]
        model = load_model(best_row)
        split_idx = load_split(task_name)
        spec = spec_for_feature_set(best_row["feature_set"])
        x_train = df.loc[split_idx["train"], spec.all_features]
        x_test = df.loc[split_idx["test"], spec.all_features]
        y_test = df.loc[split_idx["test"], project.TASKS[task_name]].reset_index(drop=True)

        feature_names = project.get_feature_names(model)
        x_test_selected = project.transform_selected(model, x_test)
        classifier = model.named_steps["classifier"]

        rng = np.random.default_rng(project.RANDOM_STATE)
        sample_n = min(800, len(x_test_selected))
        sample_idx = rng.choice(len(x_test_selected), size=sample_n, replace=False)
        x_eval = x_test_selected[sample_idx]
        y_eval = y_test.iloc[sample_idx].reset_index(drop=True)

        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer(x_eval)
        values = np.asarray(shap_values.values)
        if values.ndim == 3:
            values = values[:, :, 1]

        mean_abs = np.abs(values).mean(axis=0)
        order = np.argsort(-mean_abs)
        for rank, idx in enumerate(order[:20], start=1):
            summary_rows.append(
                {
                    "task": task_name,
                    "best_model_name": best_row["model_name"],
                    "feature": feature_names[idx],
                    "mean_abs_shap": float(mean_abs[idx]),
                    "rank": rank,
                }
            )

        plt.figure(figsize=(11, 7))
        shap.summary_plot(values, x_eval, feature_names=feature_names, max_display=20, show=False)
        write_png(PLOTS_DIR / f"{task_name}_best_model_shap_beeswarm.png")

        plt.figure(figsize=(11, 7))
        shap.summary_plot(values, x_eval, feature_names=feature_names, max_display=20, plot_type="bar", show=False)
        write_png(PLOTS_DIR / f"{task_name}_best_model_shap_bar.png")

        if hasattr(classifier, "predict_proba"):
            scores = classifier.predict_proba(x_eval)[:, 1]
        else:
            scores = classifier.decision_function(x_eval)
        top_idx = int(np.argmax(scores))

        explanation = shap.Explanation(
            values=values[top_idx],
            base_values=np.asarray(shap_values.base_values)[top_idx],
            data=x_eval[top_idx],
            feature_names=feature_names,
        )
        plt.figure(figsize=(10, 7))
        shap.plots.waterfall(explanation, max_display=20, show=False)
        write_png(PLOTS_DIR / f"{task_name}_best_model_shap_waterfall_highest_risk.png")

        top_order = np.argsort(-np.abs(values[top_idx]))[:10]
        for rank, idx in enumerate(top_order, start=1):
            local_rows.append(
                {
                    "task": task_name,
                    "case": "highest_risk_sample",
                    "predicted_score": float(scores[top_idx]),
                    "true_label": int(y_eval.iloc[top_idx]),
                    "rank": rank,
                    "feature": feature_names[idx],
                    "shap_value": float(values[top_idx, idx]),
                    "feature_value": float(x_eval[top_idx, idx]),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(local_rows)


def copy_key_files() -> None:
    copies = [
        project.INPUT_CSV,
        project.RESULTS_CSV,
        project.METHODOLOGY_JSON,
        project.selection_csv_file("injury"),
        project.selection_csv_file("fatality"),
        project.REPORTS_DIR / "dataset_feature_reference__final_tornado_dataset_region_reduced.md",
    ]
    for source in copies:
        if source.exists():
            shutil.copy2(source, FINAL_DIR / source.name)


def df_code_block(df: pd.DataFrame) -> str:
    if df.empty:
        return "```text\n<empty>\n```"
    return "```text\n" + df.to_string(index=False) + "\n```"


def write_report(
    df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    detailed_results: pd.DataFrame,
    summary_df: pd.DataFrame,
    delong_all: pd.DataFrame,
    delong_same_model: pd.DataFrame,
    delong_plot_manifest: pd.DataFrame,
    shap_global: pd.DataFrame,
    shap_local: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Final Teammate Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Final modeling dataset used: `{project.INPUT_CSV.name}`")
    lines.append(f"- Full results table: `{(TABLES_DIR / 'full_model_results_detailed.csv').name}`")
    lines.append(f"- All-pairs DeLong table: `{(TABLES_DIR / 'pairwise_delong_all_models.csv').name}`")
    lines.append(f"- Same-model feature-set DeLong table: `{(TABLES_DIR / 'pairwise_delong_same_model_feature_sets.csv').name}`")
    lines.append(f"- SHAP plot folder: `{PLOTS_DIR.name}`")
    lines.append("")
    lines.append("## Dataset Lineage")
    lines.append("")
    lines.append("- `final_tornado_dataset.csv`: original merged table after cleaning and county enrichment.")
    lines.append("- `final_tornado_dataset_region.csv`: intermediate table where `st` was replaced with `region`.")
    lines.append("- `final_tornado_dataset_region_reduced.csv`: final modeling table used in this run.")
    lines.append("- The final models were trained only on the reduced region-based dataset, not on all three CSV layers.")
    lines.append("")
    lines.append("## Dataset Summary")
    lines.append("")
    lines.append(f"- Rows: `{len(dataset_df):,}`")
    lines.append(f"- Columns: `{len(dataset_df.columns)}`")
    lines.append(f"- Reduced dataset columns are saved in `{project.INPUT_CSV.name}` and listed in `dataset_feature_reference__final_tornado_dataset_region_reduced.md`.")
    lines.append("")
    lines.append(df_code_block(summary_df))
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("### Preprocessing And Feature Design")
    lines.append("")
    lines.append("- State was replaced by broader region before modeling.")
    lines.append("- The candidate feature set was reduced before modeling to keep only PDF-aligned storm intensity/geometry, season/region, and county vulnerability/exposure variables.")
    lines.append("- Removed before modeling: raw coordinates, raw year/month/day fields, segmentation diagnostics, overlay metadata flags, and duplicate magnitude encoding.")
    lines.append("- Numeric preprocessing: median imputation then standard scaling.")
    lines.append("- Categorical preprocessing: constant-impute `Missing`, then one-hot encode with `handle_unknown='ignore'`.")
    lines.append("")
    lines.append("### Data Splits")
    lines.append("")
    lines.append(f"- Train/validation/test split: `{project.TRAIN_SHARE:.0%} / {project.VAL_SHARE:.0%} / {project.TEST_SHARE:.0%}`")
    lines.append(f"- Random state: `{project.RANDOM_STATE}`")
    lines.append("- Validation thresholds were chosen by maximizing validation F1 and then applied to the untouched test split.")
    lines.append("")
    lines.append("### Feature Selection")
    lines.append("")
    lines.append("- Feature selection was run separately for injury and fatality on the training split only.")
    lines.append("- Selector model: L1-regularized logistic regression with `solver='liblinear'`, `class_weight='balanced'`, fixed `C=0.05`, `max_iter=2000`, `tol=1e-3`.")
    lines.append("- No selector CV was used in the final simplified pass.")
    lines.append("- If more than 20 transformed features survived, they were ranked by absolute coefficient magnitude and capped to the top 20.")
    lines.append("- Injury selector output: 20 selected of 44 transformed columns.")
    lines.append("- Fatality selector output: 20 selected of 44 transformed columns.")
    lines.append("")
    lines.append("### Model Families And Tuning")
    lines.append("")
    lines.append(f"- CV folds for tuning: `{project.DEFAULT_CV_SPLITS}`")
    lines.append("- CV scoring metric for model selection: average precision (`AUPRC`).")
    lines.append("- Baseline logistic regression used `GridSearchCV` over `C in [0.01, 0.1, 1.0, 5.0, 10.0]`.")
    lines.append("- Selected-feature logistic regression used `GridSearchCV` over `C in [0.05, 0.1, 0.5, 1.0, 5.0]`.")
    lines.append("- Random forest used `RandomizedSearchCV` with 6 draws over:")
    lines.append("  `n_estimators in [300, 500]`, `max_depth in [None, 12, 20]`, `min_samples_leaf in [1, 3, 5]`, `max_features in ['sqrt', 0.5]`.")
    lines.append("- XGBoost used `RandomizedSearchCV` with 6 draws over:")
    lines.append("  `n_estimators in [200, 350]`, `max_depth in [3, 5, 7]`, `learning_rate in [0.03, 0.08, 0.15]`, `min_child_weight in [1, 3, 5]`, `subsample in [0.8, 1.0]`, `colsample_bytree in [0.7, 1.0]`.")
    lines.append("- SVM used `LinearSVC` for tractability and `GridSearchCV` over `C in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]`.")
    lines.append("- Neural network used `MLPClassifier(early_stopping=True, max_iter=300)` with `RandomizedSearchCV` over:")
    lines.append("  `hidden_layer_sizes in [(64,), (128,), (64, 32)]`, `alpha in [0.0001, 0.001, 0.01]`, `learning_rate_init in [0.001, 0.003]`.")
    lines.append("- KNN used `KNeighborsClassifier(weights='distance')` with `GridSearchCV` over `n_neighbors in [11, 21, 31, 41]` and `p in [1, 2]`.")
    lines.append("")
    lines.append("### Class Imbalance Handling")
    lines.append("")
    lines.append("- Baseline logistic: `class_weight='balanced'`.")
    lines.append("- Selected-feature logistic: `class_weight='balanced'`.")
    lines.append("- Random forest: `class_weight='balanced_subsample'`.")
    lines.append("- XGBoost: `scale_pos_weight = negatives / positives` from the training split.")
    lines.append("- SVM: `class_weight='balanced'`.")
    lines.append("- Neural network: balanced sample weights passed into `fit`.")
    lines.append("- KNN: no native class-weight support in sklearn, so distance weighting was used and no synthetic resampling was added.")
    lines.append("")
    lines.append("## Full Results")
    lines.append("")
    lines.append("The exhaustive numeric results table is saved as `tables/full_model_results_detailed.csv` and reproduced below.")
    lines.append("")
    for task_name in ["injury", "fatality"]:
        lines.append(f"### {task_name.title()} Full Metrics")
        lines.append("")
        task_df = detailed_results.loc[detailed_results["task"] == task_name, [
            "model_id",
            "cv_best_auprc",
            "validation_threshold",
            "selected_feature_count",
            "val_precision",
            "val_recall",
            "val_f1",
            "val_auroc",
            "val_auprc",
            "val_tp",
            "val_fp",
            "val_tn",
            "val_fn",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_auroc",
            "test_auprc",
            "test_tp",
            "test_fp",
            "test_tn",
            "test_fn",
        ]].copy()
        lines.append(df_code_block(task_df))
        lines.append("")
        params_df = detailed_results.loc[detailed_results["task"] == task_name, ["model_id", "best_params_json"]].copy()
        lines.append(f"### {task_name.title()} Best Hyperparameters")
        lines.append("")
        lines.append(df_code_block(params_df))
        lines.append("")
    lines.append("## DeLong Analysis")
    lines.append("")
    lines.append("Two DeLong outputs were generated:")
    lines.append("")
    lines.append("- `pairwise_delong_all_models.csv`: every pair of completed runs within each task.")
    lines.append("- `pairwise_delong_same_model_feature_sets.csv`: filtered same-model comparisons across different feature sets/runs.")
    lines.append("- `significant_delong_plot_manifest.csv`: every comparison with `p < 0.05` and its ROC PNG filename.")
    lines.append("")
    for task_name in ["injury", "fatality"]:
        lines.append(f"### {task_name.title()} All-Pairs DeLong")
        lines.append("")
        lines.append(df_code_block(delong_all.loc[delong_all["task"] == task_name]))
        lines.append("")
        filtered = delong_same_model.loc[delong_same_model["task"] == task_name]
        lines.append(f"### {task_name.title()} Same-Model Feature-Set DeLong")
        lines.append("")
        lines.append(df_code_block(filtered))
        lines.append("")
    lines.append("### Significant DeLong ROC Plot Manifest")
    lines.append("")
    lines.append(df_code_block(delong_plot_manifest))
    lines.append("")
    lines.append("## SHAP Analysis")
    lines.append("")
    lines.append("SHAP was computed for the best validation-AUPRC model in each task. In this run, the best model was XGBoost for both injury and fatality.")
    lines.append("")
    for task_name in ["injury", "fatality"]:
        lines.append(f"### {task_name.title()} SHAP Global Summary")
        lines.append("")
        lines.append(df_code_block(shap_global.loc[shap_global["task"] == task_name]))
        lines.append("")
        lines.append(f"![{task_name} SHAP beeswarm]({(PLOTS_DIR / f'{task_name}_best_model_shap_beeswarm.png').resolve().as_posix()})")
        lines.append("")
        lines.append(f"![{task_name} SHAP bar]({(PLOTS_DIR / f'{task_name}_best_model_shap_bar.png').resolve().as_posix()})")
        lines.append("")
        lines.append(f"![{task_name} SHAP waterfall]({(PLOTS_DIR / f'{task_name}_best_model_shap_waterfall_highest_risk.png').resolve().as_posix()})")
        lines.append("")
        lines.append(f"### {task_name.title()} SHAP Local Contributions")
        lines.append("")
        lines.append(df_code_block(shap_local.loc[shap_local["task"] == task_name]))
        lines.append("")
    lines.append("## Output Inventory")
    lines.append("")
    lines.append("- `master_report.md`: this report")
    lines.append("- `tables/full_model_results_detailed.csv`: every saved metric plus confusion counts and best params")
    lines.append("- `tables/pairwise_delong_all_models.csv`: exhaustive AUROC DeLong comparisons")
    lines.append("- `tables/pairwise_delong_same_model_feature_sets.csv`: same-model feature-set comparisons")
    lines.append("- `tables/significant_delong_plot_manifest.csv`: file map for significant DeLong ROC plots")
    lines.append("- `tables/dataset_summary.csv`: split sizes and class rates")
    lines.append("- `tables/shap_global_summary.csv`: ranked mean absolute SHAP values for the best model in each task")
    lines.append("- `tables/shap_local_contributions.csv`: top local SHAP contributions for the highest-risk sample in each task")
    lines.append("- `plots/*.png`: SHAP beeswarm, bar, and waterfall plots")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    dataset_df = project.load_dataset()
    results_df = load_results()
    summary_df, _ = dataset_summary(dataset_df)
    detailed_results = build_detailed_results(results_df)
    predictions = compute_test_predictions(dataset_df, results_df)
    delong_all, delong_same_model = compute_delong_tables(predictions)
    delong_plot_manifest = significant_delong_plots(predictions, delong_all)
    shap_global, shap_local = shap_outputs_for_best_models(dataset_df, results_df)

    summary_df.to_csv(TABLES_DIR / "dataset_summary.csv", index=False)
    detailed_results.to_csv(TABLES_DIR / "full_model_results_detailed.csv", index=False)
    delong_all.to_csv(TABLES_DIR / "pairwise_delong_all_models.csv", index=False)
    delong_same_model.to_csv(TABLES_DIR / "pairwise_delong_same_model_feature_sets.csv", index=False)
    delong_plot_manifest.to_csv(TABLES_DIR / "significant_delong_plot_manifest.csv", index=False)
    shap_global.to_csv(TABLES_DIR / "shap_global_summary.csv", index=False)
    shap_local.to_csv(TABLES_DIR / "shap_local_contributions.csv", index=False)
    copy_key_files()
    write_report(
        dataset_df,
        dataset_df,
        detailed_results,
        summary_df,
        delong_all,
        delong_same_model,
        delong_plot_manifest,
        shap_global,
        shap_local,
    )
    print(f"Wrote final package to: {FINAL_DIR}")


if __name__ == "__main__":
    main()
