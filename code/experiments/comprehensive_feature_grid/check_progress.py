from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_tornado_ml_project as project


EXISTING_RESULTS_CSV = PROJECT_ROOT / "final out" / "tables" / "full_model_results_detailed.csv"
ARTIFACTS_DIR = ROOT / "outputs" / "artifacts"


def main() -> int:
    existing_df = pd.read_csv(EXISTING_RESULTS_CSV)
    existing_keys = {
        (str(row.task), str(row.feature_set), str(row.model_name))
        for row in existing_df.itertuples(index=False)
    }

    all_keys = [
        (task_name, feature_set, model_name)
        for task_name in project.TASKS
        for feature_set in ["tornado_only", "county_enriched", "county_selected"]
        for model_name in project.MODEL_ORDER
    ]
    missing_keys = [key for key in all_keys if key not in existing_keys]

    completed_new = sorted(
        tuple(path.stem.split("__"))
        for path in ARTIFACTS_DIR.glob("*.json")
    )
    completed_set = set(completed_new)
    remaining = [key for key in missing_keys if key not in completed_set]

    print(f"existing_reused={len(existing_keys)}")
    print(f"missing_to_run={len(missing_keys)}")
    print(f"new_completed={len(completed_new)}")
    print(f"total_available_now={len(existing_keys) + len(completed_new)}")
    print(f"remaining_missing={len(remaining)}")
    if completed_new:
        print("completed_new_combos:")
        for key in completed_new:
            print("  " + " | ".join(key))
    if remaining:
        print("next_remaining_combos:")
        for key in remaining[:8]:
            print("  " + " | ".join(key))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
