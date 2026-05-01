# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:54:49 2026

@author: singc
"""




import pandas as pd
import requests
from pathlib import Path

API_KEY = "b174b0df9e07a8234e7495bbc2c318de2896af74"

OUT_CSV = r"C:\Users\singc\OneDrive\Documents\bruh code\4641\Project\acs_county_snapshot_2016_2020.csv"

def fetch_census_json(base_url, params):
    r = requests.get(base_url, params=params, timeout=60)
    print("\nSTATUS:", r.status_code)
    print("URL:", r.url)

    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}\nResponse text:\n{r.text[:1000]}")

    try:
        data = r.json()
    except Exception:
        raise RuntimeError(
            "Response was not valid JSON.\n"
            f"First 1000 chars:\n{r.text[:1000]}"
        )

    if not isinstance(data, list) or len(data) == 0:
        raise RuntimeError(f"Unexpected response structure:\n{data}")

    return pd.DataFrame(data[1:], columns=data[0])

# -------------------------
# 1) core ACS detailed vars
# -------------------------
core_vars = [
    "NAME",
    "B01003_001E",
    "B25001_001E",
    "B25024_010E",
    "B19013_001E",
    "B23025_003E",
    "B23025_005E",
]

core = fetch_census_json(
    "https://api.census.gov/data/2020/acs/acs5",
    {
        "get": ",".join(core_vars),
        "for": "county:*",
        "in": "state:*",
        "key": API_KEY,
    },
)

# -------------------------
# 2) age structure vars
# -------------------------
age_vars = [
    "NAME",
    "B01001_001E",
    "B01001_003E","B01001_004E","B01001_005E","B01001_006E",
    "B01001_020E","B01001_021E","B01001_022E","B01001_023E","B01001_024E","B01001_025E",
    "B01001_027E","B01001_028E","B01001_029E","B01001_030E",
    "B01001_044E","B01001_045E","B01001_046E","B01001_047E","B01001_048E","B01001_049E",
]

age = fetch_census_json(
    "https://api.census.gov/data/2020/acs/acs5",
    {
        "get": ",".join(age_vars),
        "for": "county:*",
        "in": "state:*",
        "key": API_KEY,
    },
)

# -------------------------
# 3) disability subject table
# -------------------------
dis = fetch_census_json(
    "https://api.census.gov/data/2020/acs/acs5/subject",
    {
        "get": "NAME,S1810_C01_001E,S1810_C02_001E",
        "for": "county:*",
        "in": "state:*",
        "key": API_KEY,
    },
)

# -------------------------
# 4) merge by county GEOID
# -------------------------
for df in [core, age, dis]:
    df["GEOID"] = df["state"] + df["county"]

county = core.merge(age.drop(columns=["NAME"]), on=["state", "county", "GEOID"], how="left")
county = county.merge(dis.drop(columns=["NAME"]), on=["state", "county", "GEOID"], how="left")

# numeric conversion
for c in county.columns:
    if c not in ["NAME", "state", "county", "GEOID"]:
        county[c] = pd.to_numeric(county[c], errors="coerce")

# derived features
county["county_population"] = county["B01003_001E"]
county["county_mobile_home_share"] = county["B25024_010E"] / county["B25001_001E"]
county["county_median_hh_income"] = county["B19013_001E"]
county["county_unemployment_rate"] = county["B23025_005E"] / county["B23025_003E"]
county["county_disability_rate"] = county["S1810_C02_001E"] / county["S1810_C01_001E"]

county["county_age_under18"] = (
    county["B01001_003E"] + county["B01001_004E"] + county["B01001_005E"] + county["B01001_006E"] +
    county["B01001_027E"] + county["B01001_028E"] + county["B01001_029E"] + county["B01001_030E"]
)

county["county_age_65plus"] = (
    county["B01001_020E"] + county["B01001_021E"] + county["B01001_022E"] +
    county["B01001_023E"] + county["B01001_024E"] + county["B01001_025E"] +
    county["B01001_044E"] + county["B01001_045E"] + county["B01001_046E"] +
    county["B01001_047E"] + county["B01001_048E"] + county["B01001_049E"]
)

county["county_age_under18_share"] = county["county_age_under18"] / county["B01001_001E"]
county["county_age_65plus_share"] = county["county_age_65plus"] / county["B01001_001E"]

keep_cols = [
    "GEOID", "NAME", "state", "county",
    "county_population",
    "county_mobile_home_share",
    "county_median_hh_income",
    "county_unemployment_rate",
    "county_disability_rate",
    "county_age_under18_share",
    "county_age_65plus_share",
]

out_path = Path(OUT_CSV)
out_path.parent.mkdir(parents=True, exist_ok=True)

county[keep_cols].to_csv(out_path, index=False)

print("\nSaved:", out_path)
print(county[keep_cols].head())
print("Shape:", county[keep_cols].shape)