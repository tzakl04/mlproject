# Tornado ML Project Summary

- Input dataset: `final_tornado_dataset_region_reduced.csv`
- Results table: `tornado_ml_results__final_tornado_dataset_region_reduced.csv`
- Methodology log: `methodology_decisions__final_tornado_dataset_region_reduced.json`

## Current Results

### Injury
 experiment_group     feature_set          model_name  selected_feature_count  val_auprc  test_auprc  test_auroc  test_f1
      model_suite county_selected             xgboost                      20   0.536340    0.543067    0.871518 0.521170
      model_suite county_selected      neural_network                      20   0.530458    0.545296    0.870758 0.517986
baseline_logistic county_enriched logistic_regression                      44   0.525575    0.532639    0.867775 0.507427
      model_suite county_selected logistic_regression                      20   0.524536    0.533477    0.867452 0.508095
      model_suite county_selected                 svm                      20   0.523821    0.534774    0.867386 0.506898
baseline_logistic    tornado_only logistic_regression                      18   0.519056    0.527921    0.866128 0.509868
      model_suite county_selected       random_forest                      20   0.508817    0.526771    0.868339 0.503425
      model_suite county_selected                 knn                      20   0.425529    0.449917    0.840753 0.480549

### Fatality
 experiment_group     feature_set          model_name  selected_feature_count  val_auprc  test_auprc  test_auroc  test_f1
      model_suite county_selected             xgboost                      20   0.472165    0.404038    0.923363 0.478747
      model_suite county_selected       random_forest                      20   0.454189    0.356562    0.910602 0.428571
baseline_logistic county_enriched logistic_regression                      44   0.452085    0.385243    0.926940 0.438356
      model_suite county_selected                 svm                      20   0.447944    0.386341    0.925629 0.433260
      model_suite county_selected logistic_regression                      20   0.443960    0.377654    0.926196 0.433249
      model_suite county_selected      neural_network                      20   0.438520    0.367006    0.922070 0.450331
baseline_logistic    tornado_only logistic_regression                      18   0.435372    0.399505    0.925191 0.449541
      model_suite county_selected                 knn                      20   0.369189    0.297754    0.872990 0.422658