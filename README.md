# Tornado Casualty Prediction Project: GitHub Submission Bundle

This folder is organized to be pushed directly to a GitHub repository for class submission.

## Best Way To Submit

Submit this folder as the contents of a normal GitHub repository, not as a `.zip` file.

If you already have an empty GitHub repo:

```powershell
cd "C:\path\to\github_repo_ready"
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
