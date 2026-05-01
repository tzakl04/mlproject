# Tornado ML Project Summary

- Input dataset: `final_tornado_dataset_region_reduced.csv`
- Results table: `tornado_ml_results.csv`
- Methodology log: `methodology_decisions.json`

## Current Results

### Injury
 experiment_group     feature_set          model_name  selected_feature_count  val_auprc  test_auprc  test_auroc  test_f1
baseline_logistic county_enriched logistic_regression                      44   0.525575    0.532639    0.867775 0.507427
      model_suite county_selected logistic_regression                      20   0.524536    0.533477    0.867452 0.508095
baseline_logistic    tornado_only logistic_regression                      18   0.519056    0.527921    0.866128 0.509868
      model_suite county_selected       random_forest                      20   0.508817    0.526771    0.868339 0.503425

### Fatality
 experiment_group     feature_set          model_name  selected_feature_count  val_auprc  test_auprc  test_auroc  test_f1
      model_suite county_selected       random_forest                      20   0.454189    0.356562    0.910602 0.428571
baseline_logistic county_enriched logistic_regression                      44   0.452085    0.385243    0.926940 0.438356
      model_suite county_selected logistic_regression                      20   0.443960    0.377654    0.926196 0.433249
baseline_logistic    tornado_only logistic_regression                      18   0.435372    0.399505    0.925191 0.449541