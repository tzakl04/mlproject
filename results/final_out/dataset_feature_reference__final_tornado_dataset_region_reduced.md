# Dataset And Feature Reference

## Short Answer

No. The final models were **not** trained on all three CSV layers.

- `final_tornado_dataset.csv` was the original merged table.
- `final_tornado_dataset_region.csv` was an intermediate prep table where `st` was replaced by `region`.
- `final_tornado_dataset_region_reduced.csv` was the **actual final modeling dataset** for the completed run.

Within that final reduced dataset:

- `baseline_logistic` was trained on `tornado_only` and `county_enriched`.
- The tuned model suite was trained on `county_selected`, which means the L1-selected subset of the `county_enriched` features.

## Dataset Layers

### 1. `final_tornado_dataset.csv`

Original merged table. Contains the full engineering output, including state code, raw coordinates, segmentation flags, and county overlay metadata.

Columns:

- `raw_row_id`
- `clean_row_id`
- `yr`
- `mo`
- `dy`
- `date`
- `month`
- `season`
- `season_cat`
- `st`
- `mag`
- `mag_raw`
- `mag_num`
- `mag_cat`
- `mag_missing`
- `inj`
- `fat`
- `inj_bin`
- `fat_bin`
- `slat`
- `slon`
- `elat`
- `elon`
- `elat_raw`
- `elon_raw`
- `len`
- `wid`
- `wid_raw`
- `end_missing`
- `wid_missing`
- `len_zero`
- `start_coord_invalid`
- `end_coord_invalid`
- `decade`
- `segment_nn_row`
- `segment_nn_km`
- `segment_nn_cross_state`
- `seg_source_1km`
- `seg_partner_1km`
- `possible_segment_1km`
- `seg_source_2km`
- `seg_partner_2km`
- `possible_segment_2km`
- `segment_cross_state_1km`
- `segment_cross_state_2km`
- `tornado_id`
- `start_county_geoid`
- `start_county_name`
- `start_county_population`
- `start_county_population_density_km2`
- `start_county_mobile_home_share`
- `start_county_median_hh_income`
- `start_county_unemployment_rate`
- `start_county_disability_rate`
- `start_county_age_under18_share`
- `start_county_age_65plus_share`
- `end_county_geoid`
- `end_county_name`
- `end_county_population`
- `end_county_population_density_km2`
- `end_county_mobile_home_share`
- `end_county_median_hh_income`
- `end_county_unemployment_rate`
- `end_county_disability_rate`
- `end_county_age_under18_share`
- `end_county_age_65plus_share`
- `county_overlay_method`
- `path_counties_n`
- `path_overlap_area_m2`
- `path_est_exposed_pop`
- `path_overlap_area_km2`
- `path_population_density_wavg`
- `path_mobile_home_share_wavg`
- `path_median_hh_income_wavg`
- `path_unemployment_rate_wavg`
- `path_disability_rate_wavg`
- `path_age_under18_share_wavg`
- `path_age_65plus_share_wavg`
- `path_features_present`
- `start_county_present`
- `path_overlay_or_start_fallback_present`

### 2. `final_tornado_dataset_region.csv`

Intermediate prep table. Same as the original merged table, except:

- `st` was removed
- `region` was inserted in its place

Columns:

- `raw_row_id`
- `clean_row_id`
- `yr`
- `mo`
- `dy`
- `date`
- `month`
- `season`
- `season_cat`
- `region`
- `mag`
- `mag_raw`
- `mag_num`
- `mag_cat`
- `mag_missing`
- `inj`
- `fat`
- `inj_bin`
- `fat_bin`
- `slat`
- `slon`
- `elat`
- `elon`
- `elat_raw`
- `elon_raw`
- `len`
- `wid`
- `wid_raw`
- `end_missing`
- `wid_missing`
- `len_zero`
- `start_coord_invalid`
- `end_coord_invalid`
- `decade`
- `segment_nn_row`
- `segment_nn_km`
- `segment_nn_cross_state`
- `seg_source_1km`
- `seg_partner_1km`
- `possible_segment_1km`
- `seg_source_2km`
- `seg_partner_2km`
- `possible_segment_2km`
- `segment_cross_state_1km`
- `segment_cross_state_2km`
- `tornado_id`
- `start_county_geoid`
- `start_county_name`
- `start_county_population`
- `start_county_population_density_km2`
- `start_county_mobile_home_share`
- `start_county_median_hh_income`
- `start_county_unemployment_rate`
- `start_county_disability_rate`
- `start_county_age_under18_share`
- `start_county_age_65plus_share`
- `end_county_geoid`
- `end_county_name`
- `end_county_population`
- `end_county_population_density_km2`
- `end_county_mobile_home_share`
- `end_county_median_hh_income`
- `end_county_unemployment_rate`
- `end_county_disability_rate`
- `end_county_age_under18_share`
- `end_county_age_65plus_share`
- `county_overlay_method`
- `path_counties_n`
- `path_overlap_area_m2`
- `path_est_exposed_pop`
- `path_overlap_area_km2`
- `path_population_density_wavg`
- `path_mobile_home_share_wavg`
- `path_median_hh_income_wavg`
- `path_unemployment_rate_wavg`
- `path_disability_rate_wavg`
- `path_age_under18_share_wavg`
- `path_age_65plus_share_wavg`
- `path_features_present`
- `start_county_present`
- `path_overlay_or_start_fallback_present`

### 3. `final_tornado_dataset_region_reduced.csv`

Final reduced modeling dataset. This is the one used for the finished experiment run.

Columns:

- `raw_row_id`
- `clean_row_id`
- `date`
- `season`
- `region`
- `mag_num`
- `mag_missing`
- `inj`
- `fat`
- `inj_bin`
- `fat_bin`
- `len`
- `wid`
- `end_missing`
- `wid_missing`
- `len_zero`
- `start_county_population`
- `start_county_population_density_km2`
- `start_county_mobile_home_share`
- `start_county_median_hh_income`
- `start_county_unemployment_rate`
- `start_county_disability_rate`
- `start_county_age_under18_share`
- `start_county_age_65plus_share`
- `end_county_population`
- `end_county_population_density_km2`
- `end_county_mobile_home_share`
- `end_county_median_hh_income`
- `end_county_unemployment_rate`
- `end_county_disability_rate`
- `end_county_age_under18_share`
- `end_county_age_65plus_share`
- `path_counties_n`
- `path_est_exposed_pop`
- `path_overlap_area_km2`
- `path_population_density_wavg`
- `path_mobile_home_share_wavg`
- `path_median_hh_income_wavg`
- `path_unemployment_rate_wavg`
- `path_disability_rate_wavg`
- `path_age_under18_share_wavg`
- `path_age_65plus_share_wavg`

## Model Feature Sets Used In The Final Run

### `tornado_only`

Used for baseline logistic runs only.

Numeric:

- `mag_num`
- `mag_missing`
- `len`
- `wid`
- `end_missing`
- `wid_missing`
- `len_zero`

Categorical:

- `season`
- `region`

### `county_enriched`

Used for baseline logistic runs only.

Tornado-side numeric:

- `mag_num`
- `mag_missing`
- `len`
- `wid`
- `end_missing`
- `wid_missing`
- `len_zero`

Tornado-side categorical:

- `season`
- `region`

County/path numeric:

- `start_county_population`
- `start_county_population_density_km2`
- `start_county_mobile_home_share`
- `start_county_median_hh_income`
- `start_county_unemployment_rate`
- `start_county_disability_rate`
- `start_county_age_under18_share`
- `start_county_age_65plus_share`
- `end_county_population`
- `end_county_population_density_km2`
- `end_county_mobile_home_share`
- `end_county_median_hh_income`
- `end_county_unemployment_rate`
- `end_county_disability_rate`
- `end_county_age_under18_share`
- `end_county_age_65plus_share`
- `path_counties_n`
- `path_overlap_area_km2`
- `path_est_exposed_pop`
- `path_population_density_wavg`
- `path_mobile_home_share_wavg`
- `path_median_hh_income_wavg`
- `path_unemployment_rate_wavg`
- `path_disability_rate_wavg`
- `path_age_under18_share_wavg`
- `path_age_65plus_share_wavg`

### `county_selected`

Used for the tuned model suite. This is the L1-selected subset of the transformed `county_enriched` design matrix after preprocessing and one-hot encoding.

#### Injury `county_selected` Features

- `num__mag_num`
- `cat__region_High Plains`
- `num__mag_missing`
- `cat__region_Western`
- `cat__region_Southeast`
- `num__len`
- `cat__season_Winter`
- `cat__region_Northeast`
- `cat__season_Summer`
- `num__start_county_age_65plus_share`
- `num__end_missing`
- `num__path_population_density_wavg`
- `num__path_est_exposed_pop`
- `num__path_overlap_area_km2`
- `cat__season_Spring`
- `num__start_county_disability_rate`
- `cat__region_Southern`
- `num__len_zero`
- `num__start_county_mobile_home_share`
- `num__path_unemployment_rate_wavg`

#### Fatality `county_selected` Features

- `num__mag_num`
- `cat__region_High Plains`
- `num__path_disability_rate_wavg`
- `cat__region_Northeast`
- `cat__region_Western`
- `cat__season_Winter`
- `num__mag_missing`
- `num__start_county_disability_rate`
- `num__end_county_age_65plus_share`
- `num__len`
- `num__path_age_under18_share_wavg`
- `num__end_county_disability_rate`
- `num__start_county_unemployment_rate`
- `cat__season_Spring`
- `num__end_county_unemployment_rate`
- `num__start_county_mobile_home_share`
- `cat__region_Midwest`
- `num__path_age_65plus_share_wavg`
- `num__wid`
- `num__path_est_exposed_pop`
