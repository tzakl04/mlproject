#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE_CSV = ROOT / "final_tornado_dataset.csv"
OUTPUT_CSV = ROOT / "final_tornado_dataset_region.csv"
REDUCED_OUTPUT_CSV = ROOT / "final_tornado_dataset_region_reduced.csv"


REGION_MAP = {
    "AK": "Noncontiguous",
    "HI": "Noncontiguous",
    "AL": "Southeast",
    "FL": "Southeast",
    "GA": "Southeast",
    "NC": "Southeast",
    "SC": "Southeast",
    "VA": "Southeast",
    "PR": "Southeast",
    "VI": "Southeast",
    "AR": "Southern",
    "TN": "Southern",
    "TX": "Southern",
    "LA": "Southern",
    "MS": "Southern",
    "OK": "Southern",
    "CO": "High Plains",
    "KS": "High Plains",
    "NE": "High Plains",
    "ND": "High Plains",
    "SD": "High Plains",
    "WY": "High Plains",
    "IL": "Midwest",
    "IN": "Midwest",
    "IA": "Midwest",
    "KY": "Midwest",
    "MI": "Midwest",
    "MN": "Midwest",
    "MO": "Midwest",
    "OH": "Midwest",
    "WI": "Midwest",
    "CT": "Northeast",
    "DE": "Northeast",
    "ME": "Northeast",
    "MD": "Northeast",
    "MA": "Northeast",
    "NH": "Northeast",
    "NJ": "Northeast",
    "NY": "Northeast",
    "PA": "Northeast",
    "RI": "Northeast",
    "VT": "Northeast",
    "WV": "Northeast",
    "DC": "Northeast",
    "AZ": "Western",
    "CA": "Western",
    "ID": "Western",
    "MT": "Western",
    "NV": "Western",
    "NM": "Western",
    "OR": "Western",
    "UT": "Western",
    "WA": "Western",
}


REDUCED_COLUMNS = [
    "raw_row_id",
    "clean_row_id",
    "date",
    "season",
    "region",
    "mag_num",
    "mag_missing",
    "inj",
    "fat",
    "inj_bin",
    "fat_bin",
    "len",
    "wid",
    "end_missing",
    "wid_missing",
    "len_zero",
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
    "path_est_exposed_pop",
    "path_overlap_area_km2",
    "path_population_density_wavg",
    "path_mobile_home_share_wavg",
    "path_median_hh_income_wavg",
    "path_unemployment_rate_wavg",
    "path_disability_rate_wavg",
    "path_age_under18_share_wavg",
    "path_age_65plus_share_wavg",
]


def main() -> None:
    df = pd.read_csv(SOURCE_CSV, low_memory=False)
    if "st" not in df.columns:
        raise ValueError("Expected source dataset to include an 'st' column.")

    region = df["st"].map(REGION_MAP)
    missing_codes = sorted(df.loc[region.isna(), "st"].dropna().astype(str).unique().tolist())
    if missing_codes:
        raise ValueError(f"Unmapped state/territory codes: {missing_codes}")

    insert_at = int(df.columns.get_loc("st"))
    region_df = df.drop(columns=["st"]).copy()
    region_df.insert(insert_at, "region", region.astype("string"))
    region_df.to_csv(OUTPUT_CSV, index=False)

    missing_columns = [col for col in REDUCED_COLUMNS if col not in region_df.columns]
    if missing_columns:
        raise ValueError(f"Missing reduced-dataset columns: {missing_columns}")
    reduced_df = region_df[REDUCED_COLUMNS].copy()
    reduced_df.to_csv(REDUCED_OUTPUT_CSV, index=False)

    counts = region_df["region"].value_counts(dropna=False).sort_index()
    print(f"Wrote {OUTPUT_CSV.name} with {len(region_df):,} rows and {len(region_df.columns)} columns.")
    print(
        f"Wrote {REDUCED_OUTPUT_CSV.name} with {len(reduced_df):,} rows and {len(reduced_df.columns)} columns."
    )
    print("Region counts:")
    print(counts.to_string())


if __name__ == "__main__":
    main()
