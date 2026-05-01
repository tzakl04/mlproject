#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.neighbors import BallTree
except Exception as e:  # pragma: no cover
    raise ImportError(
        "This script requires scikit-learn for BallTree-based segmentation flags. "
        "Install it with `pip install scikit-learn`."
    ) from e

# =========================
# USER-EDITABLE SETTINGS
# =========================
INPUT_CSV = "us_tornado_dataset_1950_2021.csv"
OUTPUT_CSV = "us_tornado_cleaned_all_years.csv"
SUMMARY_JSON = "us_tornado_cleaning_summary.json"
AUDIT_CSV = "us_tornado_cleaning_audit_by_decade.csv"

STRICT_SEGMENT_KM = 1.0
LOOSE_SEGMENT_KM = 2.0

EARTH_RADIUS_KM = 6371.0088


def safe_int(x):
    if pd.isna(x):
        return None
    return int(x)


def safe_float(x):
    if pd.isna(x):
        return None
    return float(x)


def month_to_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    return "Fall"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.loc[df["date"].notna()].copy()

    df["month"] = df["date"].dt.month.astype("int16")
    df["season"] = df["month"].apply(month_to_season)
    df["season_cat"] = pd.Categorical(
        df["season"], categories=["Winter", "Spring", "Summer", "Fall"], ordered=False
    )
    return df


def add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["inj_bin"] = (df["inj"] > 0).astype("int8")
    df["fat_bin"] = (df["fat"] > 0).astype("int8")
    return df


def add_missingness_and_clean_values(df: pd.DataFrame) -> pd.DataFrame:
    df["mag_raw"] = df["mag"]
    df["wid_raw"] = df["wid"]
    df["elat_raw"] = df["elat"]
    df["elon_raw"] = df["elon"]

    df["mag_missing"] = (df["mag"] == -9).astype("int8")
    df["mag_num"] = df["mag"].replace(-9, np.nan)
    df["mag_cat"] = df["mag"].replace(-9, np.nan)
    df["mag_cat"] = df["mag_cat"].apply(lambda x: "Unknown" if pd.isna(x) else str(int(x)))

    end_missing_mask = (df["elat"] == 0) & (df["elon"] == 0)
    df["end_missing"] = end_missing_mask.astype("int8")
    df.loc[end_missing_mask, ["elat", "elon"]] = np.nan

    width_missing_mask = df["wid"] == 0
    df["wid_missing"] = width_missing_mask.astype("int8")
    df.loc[width_missing_mask, "wid"] = np.nan

    df["len_zero"] = (df["len"] == 0).astype("int8")

    df["start_coord_invalid"] = (~df["slat"].between(-90, 90)) | (~df["slon"].between(-180, 180))
    df["end_coord_invalid"] = (
        (df["elat"].notna() & ~df["elat"].between(-90, 90))
        | (df["elon"].notna() & ~df["elon"].between(-180, 180))
    )
    df["start_coord_invalid"] = df["start_coord_invalid"].astype("int8")
    df["end_coord_invalid"] = df["end_coord_invalid"].astype("int8")

    bad_start = df["start_coord_invalid"] == 1
    df.loc[bad_start, ["slat", "slon"]] = np.nan
    bad_end = df["end_coord_invalid"] == 1
    df.loc[bad_end, ["elat", "elon"]] = np.nan

    return df


def _nearest_start_for_same_day(group: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=group.index)
    out["segment_nn_row"] = pd.Series(pd.NA, index=group.index, dtype="Int64")
    out["segment_nn_km"] = np.nan
    out["segment_nn_cross_state"] = pd.Series(0, index=group.index, dtype="int8")

    valid_start = group["slat"].notna() & group["slon"].notna()
    valid_end = group["elat"].notna() & group["elon"].notna()

    start_rows = group.loc[valid_start]
    end_rows = group.loc[valid_end]

    if len(start_rows) == 0 or len(end_rows) == 0:
        return out

    start_coords = np.deg2rad(start_rows[["slat", "slon"]].to_numpy())
    end_coords = np.deg2rad(end_rows[["elat", "elon"]].to_numpy())

    tree = BallTree(start_coords, metric="haversine")
    k = 2 if len(start_rows) > 1 else 1
    dists_rad, inds = tree.query(end_coords, k=k)

    for i, end_idx in enumerate(end_rows.index):
        chosen = None
        for dist_rad, start_pos in zip(np.atleast_1d(dists_rad[i]), np.atleast_1d(inds[i])):
            start_idx = start_rows.index[start_pos]
            if start_idx != end_idx:
                chosen = (start_idx, dist_rad * EARTH_RADIUS_KM)
                break

        if chosen is None:
            continue

        partner_idx, dist_km = chosen
        out.at[end_idx, "segment_nn_row"] = int(partner_idx)
        out.at[end_idx, "segment_nn_km"] = float(dist_km)
        out.at[end_idx, "segment_nn_cross_state"] = int(group.at[end_idx, "st"] != group.at[partner_idx, "st"])

    return out


def add_segmentation_flags(df: pd.DataFrame, strict_km: float = 1.0, loose_km: float = 2.0) -> pd.DataFrame:
    df["segment_nn_row"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df["segment_nn_km"] = np.nan
    df["segment_nn_cross_state"] = pd.Series(0, index=df.index, dtype="int8")

    nearest_parts = []
    for _, idx in df.groupby("date").groups.items():
        group = df.loc[idx]
        nearest_parts.append(_nearest_start_for_same_day(group))

    if nearest_parts:
        nearest = pd.concat(nearest_parts).sort_index()
        for col in ["segment_nn_row", "segment_nn_km", "segment_nn_cross_state"]:
            df.loc[nearest.index, col] = nearest[col]

    strict_source_mask = df["segment_nn_km"].notna() & (df["segment_nn_km"] <= strict_km)
    loose_source_mask = df["segment_nn_km"].notna() & (df["segment_nn_km"] <= loose_km)

    df["seg_source_1km"] = strict_source_mask.astype("int8")
    df["seg_source_2km"] = loose_source_mask.astype("int8")

    strict_partner_idx = df.loc[strict_source_mask, "segment_nn_row"].dropna().astype(int).unique()
    loose_partner_idx = df.loc[loose_source_mask, "segment_nn_row"].dropna().astype(int).unique()

    df["seg_partner_1km"] = 0
    df["seg_partner_2km"] = 0
    if len(strict_partner_idx) > 0:
        df.loc[strict_partner_idx, "seg_partner_1km"] = 1
    if len(loose_partner_idx) > 0:
        df.loc[loose_partner_idx, "seg_partner_2km"] = 1

    df["seg_partner_1km"] = df["seg_partner_1km"].astype("int8")
    df["seg_partner_2km"] = df["seg_partner_2km"].astype("int8")

    df["possible_segment_1km"] = ((df["seg_source_1km"] == 1) | (df["seg_partner_1km"] == 1)).astype("int8")
    df["possible_segment_2km"] = ((df["seg_source_2km"] == 1) | (df["seg_partner_2km"] == 1)).astype("int8")

    df["segment_cross_state_1km"] = (
        (df["seg_source_1km"] == 1) & (df["segment_nn_cross_state"] == 1)
    ).astype("int8")
    df["segment_cross_state_2km"] = (
        (df["seg_source_2km"] == 1) & (df["segment_nn_cross_state"] == 1)
    ).astype("int8")

    return df


def build_decade_audit(df: pd.DataFrame) -> pd.DataFrame:
    audit = (
        df.groupby("decade", dropna=False)
        .agg(
            rows=("yr", "size"),
            injury_positive=("inj_bin", "sum"),
            fatality_positive=("fat_bin", "sum"),
            injury_positive_rate=("inj_bin", "mean"),
            fatality_positive_rate=("fat_bin", "mean"),
            end_missing_rate=("end_missing", "mean"),
            mag_missing_rate=("mag_missing", "mean"),
            wid_missing_rate=("wid_missing", "mean"),
            len_zero_rate=("len_zero", "mean"),
            possible_segment_1km_rate=("possible_segment_1km", "mean"),
            possible_segment_2km_rate=("possible_segment_2km", "mean"),
        )
        .reset_index()
        .sort_values("decade")
    )
    return audit


def clean_dataset(input_csv: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    raw = pd.read_csv(input_csv)
    raw_rows = len(raw)

    raw["raw_row_id"] = np.arange(len(raw), dtype=int)

    dedupe_cols = [c for c in raw.columns if c != "raw_row_id"]
    duplicate_rows = int(raw.duplicated(subset=dedupe_cols).sum())
    df = raw.drop_duplicates(subset=dedupe_cols).copy().reset_index(drop=True)
    df["clean_row_id"] = np.arange(len(df), dtype=int)

    pre_parse_rows = len(df)
    df = add_time_features(df)
    invalid_dates_dropped = pre_parse_rows - len(df)

    df = add_target_columns(df)
    df = add_missingness_and_clean_values(df)
    df["decade"] = ((df["yr"] // 10) * 10).astype(int)
    df = add_segmentation_flags(df, strict_km=STRICT_SEGMENT_KM, loose_km=LOOSE_SEGMENT_KM)

    preferred_order = [
        "raw_row_id", "clean_row_id",
        "yr", "mo", "dy", "date", "month", "season", "season_cat", "st",
        "mag", "mag_raw", "mag_num", "mag_cat", "mag_missing",
        "inj", "fat", "inj_bin", "fat_bin",
        "slat", "slon", "elat", "elon", "elat_raw", "elon_raw",
        "len", "wid", "wid_raw",
        "end_missing", "wid_missing", "len_zero",
        "start_coord_invalid", "end_coord_invalid",
        "decade",
        "segment_nn_row", "segment_nn_km", "segment_nn_cross_state",
        "seg_source_1km", "seg_partner_1km", "possible_segment_1km",
        "seg_source_2km", "seg_partner_2km", "possible_segment_2km",
        "segment_cross_state_1km", "segment_cross_state_2km",
    ]
    existing = [c for c in preferred_order if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    df = df[existing + remaining]

    summary = {
        "raw_rows": int(raw_rows),
        "duplicate_rows_removed": int(duplicate_rows),
        "invalid_dates_dropped": int(invalid_dates_dropped),
        "clean_rows": int(len(df)),
        "year_min": safe_int(df["yr"].min()),
        "year_max": safe_int(df["yr"].max()),
        "states_unique": int(df["st"].nunique(dropna=True)),
        "injury_positive_rows": int(df["inj_bin"].sum()),
        "fatality_positive_rows": int(df["fat_bin"].sum()),
        "mag_missing_rows": int(df["mag_missing"].sum()),
        "end_missing_rows": int(df["end_missing"].sum()),
        "wid_missing_rows": int(df["wid_missing"].sum()),
        "len_zero_rows": int(df["len_zero"].sum()),
        "possible_segment_1km_rows": int(df["possible_segment_1km"].sum()),
        "possible_segment_2km_rows": int(df["possible_segment_2km"].sum()),
        "segment_cross_state_1km_source_rows": int(df["segment_cross_state_1km"].sum()),
        "segment_cross_state_2km_source_rows": int(df["segment_cross_state_2km"].sum()),
        "notes": [
            "All years are retained.",
            "No imputation is performed here. NaNs are left in place so that imputation can be fit on training data only later.",
            "Likely track segmentation is flagged but not merged. The unit remains the report row, not a guaranteed unique physical tornado.",
            "Magnitude -9 is treated as unknown; width 0 is treated as coded missingness; endpoint (0,0) is treated as coded missingness; length 0 is retained and flagged.",
            "Season is derived from month as Winter/Spring/Summer/Fall.",
        ],
    }

    audit = build_decade_audit(df)
    return df, summary, audit


def main():
    df, summary, audit = clean_dataset(INPUT_CSV)

    output_path = Path(OUTPUT_CSV)
    summary_path = Path(SUMMARY_JSON)
    audit_path = Path(AUDIT_CSV)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    audit.to_csv(audit_path, index=False)

    print("Saved cleaned data to:", output_path)
    print("Saved summary JSON to:", summary_path)
    print("Saved per-decade audit CSV to:", audit_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
