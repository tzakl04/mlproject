from __future__ import annotations

import csv
import shutil
import textwrap
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
BUNDLE_ROOT = PROJECT_ROOT / "github_repo_ready"

DOWNLOADS_PDF = Path.home() / "Downloads" / "ML_Project (1).pdf"
WORKSPACE_PDF = PROJECT_ROOT / "ML_Project.pdf"


REQUIREMENTS_TEXT = """\
geopandas
joblib
markdown
matplotlib
numpy
openpyxl
pandas
requests
scikit-learn
scipy
shap
shapely
xgboost
"""


def ensure_clean_bundle_root() -> None:
    if BUNDLE_ROOT.exists():
        if BUNDLE_ROOT.name != "github_repo_ready":
            raise RuntimeError(f"Refusing to remove unexpected directory: {BUNDLE_ROOT}")
        shutil.rmtree(BUNDLE_ROOT)
    BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst_relative: str) -> None:
    dst = BUNDLE_ROOT / dst_relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst_relative: str, exclude_dirs: set[str] | None = None, exclude_files: set[str] | None = None) -> None:
    exclude_dirs = exclude_dirs or set()
    exclude_files = exclude_files or set()
    dst_root = BUNDLE_ROOT / dst_relative
    for path in src.rglob("*"):
        relative = path.relative_to(src)
        if any(part in exclude_dirs for part in relative.parts):
            continue
        if path.is_file() and path.name in exclude_files:
            continue
        dst = dst_root / relative
        if path.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)


def zip_tree(src_dir: Path, dst_relative_zip: str) -> None:
    dst_zip = BUNDLE_ROOT / dst_relative_zip
    dst_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dst_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                arcname = Path(src_dir.name) / path.relative_to(src_dir)
                zf.write(path, arcname.as_posix())


def build_readme() -> str:
    return textwrap.dedent(
        """\
        # Tornado Casualty Prediction Project: GitHub Submission Bundle

        This folder is organized to be pushed directly to a GitHub repository for class submission.

        ## Best Way To Submit

        Submit this folder as the contents of a normal GitHub repository, not as a `.zip` file.

        If you already have an empty GitHub repo:

        ```powershell
        cd "C:\\path\\to\\github_repo_ready"
        git init
        git add .
        git commit -m "Initial project submission"
        git branch -M main
        git remote add origin <your-repo-url>
        git push -u origin main
        ```

        If you prefer the GitHub web UI, open your empty repository in the browser and drag the contents of this folder into the repo root.

        ## Folder Structure

        - `paper/`: project paper PDFs.
        - `code/`: data-prep, modeling, reporting, and experiment scripts.
        - `data/raw/`: original raw inputs and a zipped county shapefile bundle.
        - `data/processed/`: cleaned/intermediate/final modeling tables.
        - `results/`: final reports, plots, tables, and experiment outputs.
        - `notes/`: submission manifest and file inventory.

        ## Important Notes

        - The county shapefile is stored as `data/raw/tl_2020_us_county.zip` to stay GitHub-friendly. Unzip it before rerunning the spatial join pipeline.
        - Serialized trained model binaries were intentionally omitted from this bundle:
          - `analysis_outputs/models/`
          - `comprehensive_feature_grid/outputs/models/`
          - `isolated_downsample_xgboost/outputs/models/`
        - Those omitted files are reproducible from the included scripts and datasets, and one of them exceeded GitHub's 100 MB single-file limit.

        ## Main Entry Points

        - Data cleaning: `code/data_prep/tornado_cleaning_pipeline_v2.py`
        - County enrichment / dataset build: `code/data_prep/build_final_tornado_dataset_excel_v3.py`
        - Region-reduced dataset build: `code/data_prep/build_region_tornado_dataset.py`
        - Main modeling pipeline: `code/modeling/run_tornado_ml_project.py`
        - Comprehensive appendix grid: `code/experiments/comprehensive_feature_grid/run_comprehensive_feature_grid.py`
        - Final packaged report assets: `results/final_out/`
        """
    )


def build_manifest(total_files: int, total_size_mb: float, max_file_rel: str, max_file_size_mb: float) -> str:
    return textwrap.dedent(
        f"""\
        # Submission Manifest

        This bundle was generated from the working project folder to create a cleaner repo-ready submission package.

        ## Included

        - Core data-preparation scripts
        - Modeling and evaluation scripts
        - Final processed datasets
        - Final report assets, plots, and tables
        - Comprehensive appendix-grid results
        - Downsampled XGBoost side experiment
        - Two paper PDF snapshots

        ## Intentionally Omitted

        - `analysis_outputs/models/`
        - `comprehensive_feature_grid/outputs/models/`
        - `isolated_downsample_xgboost/outputs/models/`
        - `__pycache__/`

        Reason: serialized model binaries are reproducible, add substantial repo weight, and at least one file exceeds GitHub's standard single-file upload limit.

        ## Bundle Stats

        - Total files: {total_files}
        - Total size: {total_size_mb:.2f} MB
        - Largest file: `{max_file_rel}` ({max_file_size_mb:.2f} MB)
        """
    )


def write_text(dst_relative: str, contents: str) -> None:
    dst = BUNDLE_ROOT / dst_relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(contents, encoding="utf-8")


def write_inventory() -> tuple[int, float, str, float]:
    rows: list[tuple[str, int, float]] = []
    for path in sorted(BUNDLE_ROOT.rglob("*")):
        if path.is_file():
            rel = path.relative_to(BUNDLE_ROOT).as_posix()
            size_bytes = path.stat().st_size
            rows.append((rel, size_bytes, size_bytes / (1024 * 1024)))

    inventory_path = BUNDLE_ROOT / "notes" / "repository_inventory.csv"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    with inventory_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["relative_path", "size_bytes", "size_mb"])
        for rel, size_bytes, size_mb in rows:
            writer.writerow([rel, size_bytes, f"{size_mb:.4f}"])

    total_files = len(rows)
    total_size_mb = sum(size_mb for _, _, size_mb in rows)
    max_rel, max_bytes, max_mb = max(rows, key=lambda row: row[1])
    return total_files, total_size_mb, max_rel, max_mb


def main() -> int:
    ensure_clean_bundle_root()

    # Paper snapshots
    if DOWNLOADS_PDF.exists():
        copy_file(DOWNLOADS_PDF, "paper/ML_Project_latest_from_downloads.pdf")
    if WORKSPACE_PDF.exists():
        copy_file(WORKSPACE_PDF, "paper/ML_Project_workspace_snapshot.pdf")

    # Code
    copy_file(PROJECT_ROOT / "tornado_cleaning_pipeline_v2.py", "code/data_prep/tornado_cleaning_pipeline_v2.py")
    copy_file(PROJECT_ROOT / "api_census.py", "code/data_prep/api_census.py")
    copy_file(PROJECT_ROOT / "build_final_tornado_dataset_excel_v3.py", "code/data_prep/build_final_tornado_dataset_excel_v3.py")
    copy_file(PROJECT_ROOT / "build_region_tornado_dataset.py", "code/data_prep/build_region_tornado_dataset.py")
    copy_file(PROJECT_ROOT / "run_tornado_ml_project.py", "code/modeling/run_tornado_ml_project.py")
    copy_file(PROJECT_ROOT / "build_final_out_package.py", "code/reporting/build_final_out_package.py")
    copy_file(PROJECT_ROOT / "export_master_report_to_docx.py", "code/reporting/export_master_report_to_docx.py")
    copy_file(PROJECT_ROOT / "comprehensive_feature_grid" / "run_comprehensive_feature_grid.py", "code/experiments/comprehensive_feature_grid/run_comprehensive_feature_grid.py")
    copy_file(PROJECT_ROOT / "comprehensive_feature_grid" / "check_progress.py", "code/experiments/comprehensive_feature_grid/check_progress.py")
    copy_file(PROJECT_ROOT / "isolated_downsample_xgboost" / "run_downsampled_xgboost.py", "code/experiments/isolated_downsample_xgboost/run_downsampled_xgboost.py")
    copy_file(Path(__file__), "code/utilities/prepare_github_repo_ready.py")

    # Raw data
    copy_file(PROJECT_ROOT / "us_tornado_dataset_1950_2021.csv", "data/raw/us_tornado_dataset_1950_2021.csv")
    copy_file(PROJECT_ROOT / "acs_county_snapshot_2016_2020.csv", "data/raw/acs_county_snapshot_2016_2020.csv")
    zip_tree(PROJECT_ROOT / "tl_2020_us_county", "data/raw/tl_2020_us_county.zip")

    # Processed data
    copy_file(PROJECT_ROOT / "us_tornado_cleaned_all_years.csv", "data/processed/us_tornado_cleaned_all_years.csv")
    copy_file(PROJECT_ROOT / "us_tornado_cleaning_summary.json", "data/processed/us_tornado_cleaning_summary.json")
    copy_file(PROJECT_ROOT / "us_tornado_cleaning_audit_by_decade.csv", "data/processed/us_tornado_cleaning_audit_by_decade.csv")
    copy_file(PROJECT_ROOT / "tornado_county_bridge.csv", "data/processed/tornado_county_bridge.csv")
    copy_file(PROJECT_ROOT / "final_tornado_dataset.csv", "data/processed/final_tornado_dataset.csv")
    copy_file(PROJECT_ROOT / "final_tornado_dataset.xlsx", "data/processed/final_tornado_dataset.xlsx")
    copy_file(PROJECT_ROOT / "final_tornado_dataset_region.csv", "data/processed/final_tornado_dataset_region.csv")
    copy_file(PROJECT_ROOT / "final_tornado_dataset_region_reduced.csv", "data/processed/final_tornado_dataset_region_reduced.csv")

    # Results
    copy_tree(
        PROJECT_ROOT / "analysis_outputs" / "artifacts",
        "results/analysis_outputs/artifacts",
    )
    copy_tree(
        PROJECT_ROOT / "analysis_outputs" / "reports",
        "results/analysis_outputs/reports",
    )
    copy_tree(
        PROJECT_ROOT / "analysis_outputs" / "splits",
        "results/analysis_outputs/splits",
    )

    copy_tree(
        PROJECT_ROOT / "final out",
        "results/final_out",
        exclude_files={"final_tornado_dataset_region_reduced.csv"},
    )

    copy_tree(
        PROJECT_ROOT / "comprehensive_feature_grid" / "outputs" / "artifacts",
        "results/comprehensive_feature_grid/outputs/artifacts",
    )
    copy_file(
        PROJECT_ROOT / "comprehensive_feature_grid" / "outputs" / "appendix_comprehensive_feature_grid_tables.tex",
        "results/comprehensive_feature_grid/outputs/appendix_comprehensive_feature_grid_tables.tex",
    )
    copy_file(
        PROJECT_ROOT / "comprehensive_feature_grid" / "outputs" / "comprehensive_feature_grid_results.csv",
        "results/comprehensive_feature_grid/outputs/comprehensive_feature_grid_results.csv",
    )
    copy_file(
        PROJECT_ROOT / "comprehensive_feature_grid" / "outputs" / "comprehensive_feature_grid_summary.json",
        "results/comprehensive_feature_grid/outputs/comprehensive_feature_grid_summary.json",
    )

    copy_tree(
        PROJECT_ROOT / "isolated_downsample_xgboost" / "outputs" / "artifacts",
        "results/isolated_downsample_xgboost/outputs/artifacts",
    )
    copy_file(
        PROJECT_ROOT / "isolated_downsample_xgboost" / "outputs" / "comparison_vs_original_weighted_xgboost.csv",
        "results/isolated_downsample_xgboost/outputs/comparison_vs_original_weighted_xgboost.csv",
    )
    copy_file(
        PROJECT_ROOT / "isolated_downsample_xgboost" / "outputs" / "downsampled_xgboost_results.csv",
        "results/isolated_downsample_xgboost/outputs/downsampled_xgboost_results.csv",
    )
    copy_file(
        PROJECT_ROOT / "isolated_downsample_xgboost" / "outputs" / "downsampled_xgboost_summary.json",
        "results/isolated_downsample_xgboost/outputs/downsampled_xgboost_summary.json",
    )

    # Repo metadata
    write_text("README.md", build_readme())
    write_text("requirements.txt", REQUIREMENTS_TEXT)
    write_text(
        ".gitignore",
        "__pycache__/\n*.pyc\n*.pyo\n.DS_Store\nThumbs.db\n",
    )

    total_files, total_size_mb, max_file_rel, max_file_size_mb = write_inventory()
    write_text(
        "notes/submission_manifest.md",
        build_manifest(total_files, total_size_mb, max_file_rel, max_file_size_mb),
    )

    print(f"Created bundle at: {BUNDLE_ROOT}")
    print(f"Total files: {total_files}")
    print(f"Total size: {total_size_mb:.2f} MB")
    print(f"Largest file: {max_file_rel} ({max_file_size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
