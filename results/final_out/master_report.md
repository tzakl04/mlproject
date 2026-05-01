# Final Teammate Report

## Scope

- Final modeling dataset used: `final_tornado_dataset_region_reduced.csv`
- Full results table: `full_model_results_detailed.csv`
- All-pairs DeLong table: `pairwise_delong_all_models.csv`
- Same-model feature-set DeLong table: `pairwise_delong_same_model_feature_sets.csv`
- SHAP plot folder: `plots`

## Dataset Lineage

- `final_tornado_dataset.csv`: original merged table after cleaning and county enrichment.
- `final_tornado_dataset_region.csv`: intermediate table where `st` was replaced with `region`.
- `final_tornado_dataset_region_reduced.csv`: final modeling table used in this run.
- The final models were trained only on the reduced region-based dataset, not on all three CSV layers.

## Dataset Summary

- Rows: `67,096`
- Columns: `42`
- Reduced dataset columns are saved in `final_tornado_dataset_region_reduced.csv` and listed in `dataset_feature_reference__final_tornado_dataset_region_reduced.md`.

```text
    task  subset  rows  positive_rows  negative_rows  positive_rate
  injury overall 67096           7686          59410       0.114552
  injury   train 46966           5380          41586       0.114551
  injury     val 10065           1153           8912       0.114555
  injury    test 10065           1153           8912       0.114555
fatality overall 67096           1559          65537       0.023235
fatality   train 46966           1091          45875       0.023230
fatality     val 10065            234           9831       0.023249
fatality    test 10065            234           9831       0.023249
```

## Methodology

### Preprocessing And Feature Design

- State was replaced by broader region before modeling.
- The candidate feature set was reduced before modeling to keep only PDF-aligned storm intensity/geometry, season/region, and county vulnerability/exposure variables.
- Removed before modeling: raw coordinates, raw year/month/day fields, segmentation diagnostics, overlay metadata flags, and duplicate magnitude encoding.
- Numeric preprocessing: median imputation then standard scaling.
- Categorical preprocessing: constant-impute `Missing`, then one-hot encode with `handle_unknown='ignore'`.

### Data Splits

- Train/validation/test split: `70% / 15% / 15%`
- Random state: `42`
- Validation thresholds were chosen by maximizing validation F1 and then applied to the untouched test split.

### Feature Selection

- Feature selection was run separately for injury and fatality on the training split only.
- Selector model: L1-regularized logistic regression with `solver='liblinear'`, `class_weight='balanced'`, fixed `C=0.05`, `max_iter=2000`, `tol=1e-3`.
- No selector CV was used in the final simplified pass.
- If more than 20 transformed features survived, they were ranked by absolute coefficient magnitude and capped to the top 20.
- Injury selector output: 20 selected of 44 transformed columns.
- Fatality selector output: 20 selected of 44 transformed columns.

### Model Families And Tuning

- CV folds for tuning: `3`
- CV scoring metric for model selection: average precision (`AUPRC`).
- Baseline logistic regression used `GridSearchCV` over `C in [0.01, 0.1, 1.0, 5.0, 10.0]`.
- Selected-feature logistic regression used `GridSearchCV` over `C in [0.05, 0.1, 0.5, 1.0, 5.0]`.
- Random forest used `RandomizedSearchCV` with 6 draws over:
  `n_estimators in [300, 500]`, `max_depth in [None, 12, 20]`, `min_samples_leaf in [1, 3, 5]`, `max_features in ['sqrt', 0.5]`.
- XGBoost used `RandomizedSearchCV` with 6 draws over:
  `n_estimators in [200, 350]`, `max_depth in [3, 5, 7]`, `learning_rate in [0.03, 0.08, 0.15]`, `min_child_weight in [1, 3, 5]`, `subsample in [0.8, 1.0]`, `colsample_bytree in [0.7, 1.0]`.
- SVM used `LinearSVC` for tractability and `GridSearchCV` over `C in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]`.
- Neural network used `MLPClassifier(early_stopping=True, max_iter=300)` with `RandomizedSearchCV` over:
  `hidden_layer_sizes in [(64,), (128,), (64, 32)]`, `alpha in [0.0001, 0.001, 0.01]`, `learning_rate_init in [0.001, 0.003]`.
- KNN used `KNeighborsClassifier(weights='distance')` with `GridSearchCV` over `n_neighbors in [11, 21, 31, 41]` and `p in [1, 2]`.

### Class Imbalance Handling

- Baseline logistic: `class_weight='balanced'`.
- Selected-feature logistic: `class_weight='balanced'`.
- Random forest: `class_weight='balanced_subsample'`.
- XGBoost: `scale_pos_weight = negatives / positives` from the training split.
- SVM: `class_weight='balanced'`.
- Neural network: balanced sample weights passed into `fit`.
- KNN: no native class-weight support in sklearn, so distance weighting was used and no synthetic resampling was added.

## Full Results

The exhaustive numeric results table is saved as `tables/full_model_results_detailed.csv` and reproduced below.

### Injury Full Metrics

```text
                                             model_id  cv_best_auprc  validation_threshold  selected_feature_count  val_precision  val_recall   val_f1  val_auroc  val_auprc  val_tp  val_fp  val_tn  val_fn  test_precision  test_recall  test_f1  test_auroc  test_auprc  test_tp  test_fp  test_tn  test_fn
                  model_suite|county_selected|xgboost       0.535449              0.708093                      20       0.472626    0.591500 0.525424   0.869083   0.536340     682     761    8151     471        0.468512     0.587164 0.521170    0.871518    0.543067      677      768     8144      476
           model_suite|county_selected|neural_network       0.526944              0.725433                      20       0.476534    0.572420 0.520095   0.868620   0.530458     660     725    8187     493        0.480356     0.562012 0.517986    0.870758    0.545296      648      701     8211      505
baseline_logistic|county_enriched|logistic_regression       0.518164              0.742122                      44       0.474852    0.556808 0.512575   0.866213   0.525575     642     710    8202     511        0.472347     0.548135 0.507427    0.867775    0.532639      632      706     8206      521
      model_suite|county_selected|logistic_regression       0.518794              0.758357                      20       0.493259    0.539462 0.515327   0.865910   0.524536     622     639    8273     531        0.487261     0.530789 0.508095    0.867452    0.533477      612      644     8268      541
                      model_suite|county_selected|svm       0.517220              0.385065                      20       0.470504    0.567216 0.514353   0.865632   0.523821     654     736    8176     499        0.464595     0.557676 0.506898    0.867386    0.534774      643      741     8171      510
   baseline_logistic|tornado_only|logistic_regression       0.512388              0.756791                      18       0.480560    0.535993 0.506765   0.861290   0.519056     618     668    8244     535        0.484754     0.537728 0.509868    0.866128    0.527921      620      659     8253      533
            model_suite|county_selected|random_forest       0.516716              0.637697                      20       0.504610    0.522116 0.513214   0.864277   0.508817     602     591    8321     551        0.497041     0.509974 0.503425    0.868339    0.526771      588      595     8317      565
                      model_suite|county_selected|knn       0.426561              0.189649                      20       0.380362    0.638335 0.476684   0.841232   0.425529     736    1199    7713     417        0.385624     0.637467 0.480549    0.840753    0.449917      735     1171     7741      418
```

### Injury Best Hyperparameters

```text
                                             model_id                                                                                                                                                                                       best_params_json
                  model_suite|county_selected|xgboost {"classifier__colsample_bytree": 1.0, "classifier__learning_rate": 0.03, "classifier__max_depth": 3, "classifier__min_child_weight": 1, "classifier__n_estimators": 350, "classifier__subsample": 1.0}
           model_suite|county_selected|neural_network                                                                                         {"classifier__alpha": 0.001, "classifier__hidden_layer_sizes": [128], "classifier__learning_rate_init": 0.001}
baseline_logistic|county_enriched|logistic_regression                                                                                                                                                                                {"classifier__C": 0.01}
      model_suite|county_selected|logistic_regression                                                                                                                                                                                 {"classifier__C": 1.0}
                      model_suite|county_selected|svm                                                                                                                                                                                {"classifier__C": 0.01}
   baseline_logistic|tornado_only|logistic_regression                                                                                                                                                                                 {"classifier__C": 1.0}
            model_suite|county_selected|random_forest                                                                  {"classifier__max_depth": 12, "classifier__max_features": "sqrt", "classifier__min_samples_leaf": 5, "classifier__n_estimators": 300}
                      model_suite|county_selected|knn                                                                                                                                                    {"classifier__n_neighbors": 41, "classifier__p": 2}
```

### Fatality Full Metrics

```text
                                             model_id  cv_best_auprc  validation_threshold  selected_feature_count  val_precision  val_recall   val_f1  val_auroc  val_auprc  val_tp  val_fp  val_tn  val_fn  test_precision  test_recall  test_f1  test_auroc  test_auprc  test_tp  test_fp  test_tn  test_fn
                  model_suite|county_selected|xgboost       0.431516              0.930166                      20       0.545024    0.491453 0.516854   0.930484   0.472165     115      96    9735     119        0.502347     0.457265 0.478747    0.923363    0.404038      107      106     9725      127
            model_suite|county_selected|random_forest       0.416577              0.541918                      20       0.469697    0.529915 0.497992   0.921350   0.454189     124     140    9691     110        0.421488     0.435897 0.428571    0.910602    0.356562      102      140     9691      132
baseline_logistic|county_enriched|logistic_regression       0.423028              0.943044                      44       0.504464    0.482906 0.493450   0.931344   0.452085     113     111    9720     121        0.470588     0.410256 0.438356    0.926940    0.385243       96      108     9723      138
                      model_suite|county_selected|svm       0.428897              0.899759                      20       0.483051    0.487179 0.485106   0.930309   0.447944     114     122    9709     120        0.443946     0.423077 0.433260    0.925629    0.386341       99      124     9707      135
      model_suite|county_selected|logistic_regression       0.427542              0.958239                      20       0.575581    0.423077 0.487685   0.930140   0.443960      99      73    9758     135        0.527607     0.367521 0.433249    0.926196    0.377654       86       77     9754      148
           model_suite|county_selected|neural_network       0.429016              0.912443                      20       0.517391    0.508547 0.512931   0.930499   0.438520     119     111    9720     115        0.465753     0.435897 0.450331    0.922070    0.367006      102      117     9714      132
   baseline_logistic|tornado_only|logistic_regression       0.425114              0.947893                      18       0.509174    0.474359 0.491150   0.929956   0.435372     111     107    9724     123        0.485149     0.418803 0.449541    0.925191    0.399505       98      104     9727      136
                      model_suite|county_selected|knn       0.339582              0.202980                      20       0.463415    0.487179 0.475000   0.884518   0.369189     114     132    9699     120        0.431111     0.414530 0.422658    0.872990    0.297754       97      128     9703      137
```

### Fatality Best Hyperparameters

```text
                                             model_id                                                                                                                                                                                       best_params_json
                  model_suite|county_selected|xgboost {"classifier__colsample_bytree": 1.0, "classifier__learning_rate": 0.03, "classifier__max_depth": 3, "classifier__min_child_weight": 1, "classifier__n_estimators": 350, "classifier__subsample": 1.0}
            model_suite|county_selected|random_forest                                                                     {"classifier__max_depth": 20, "classifier__max_features": 0.5, "classifier__min_samples_leaf": 5, "classifier__n_estimators": 500}
baseline_logistic|county_enriched|logistic_regression                                                                                                                                                                                {"classifier__C": 0.01}
                      model_suite|county_selected|svm                                                                                                                                                                                {"classifier__C": 0.01}
      model_suite|county_selected|logistic_regression                                                                                                                                                                                {"classifier__C": 0.05}
           model_suite|county_selected|neural_network                                                                                         {"classifier__alpha": 0.001, "classifier__hidden_layer_sizes": [128], "classifier__learning_rate_init": 0.001}
   baseline_logistic|tornado_only|logistic_regression                                                                                                                                                                                 {"classifier__C": 5.0}
                      model_suite|county_selected|knn                                                                                                                                                    {"classifier__n_neighbors": 41, "classifier__p": 2}
```

## DeLong Analysis

Two DeLong outputs were generated:

- `pairwise_delong_all_models.csv`: every pair of completed runs within each task.
- `pairwise_delong_same_model_feature_sets.csv`: filtered same-model comparisons across different feature sets/runs.
- `significant_delong_plot_manifest.csv`: every comparison with `p < 0.05` and its ROC PNG filename.

### Injury All-Pairs DeLong

```text
  task                                            model_1_id model_1_experiment_group model_1_feature_set        model_1_name                                         model_2_id model_2_experiment_group model_2_feature_set        model_2_name    auc_1    auc_2  auc_diff        z      p_value  same_model_name  same_feature_set  same_experiment_group
injury                       model_suite|county_selected|knn              model_suite     county_selected                 knn                model_suite|county_selected|xgboost              model_suite     county_selected             xgboost 0.840753 0.871518 -0.030765 8.006955 1.110223e-15            False              True                   True
injury                       model_suite|county_selected|knn              model_suite     county_selected                 knn         model_suite|county_selected|neural_network              model_suite     county_selected      neural_network 0.840753 0.870758 -0.030005 7.851934 3.996803e-15            False              True                   True
injury                       model_suite|county_selected|knn              model_suite     county_selected                 knn          model_suite|county_selected|random_forest              model_suite     county_selected       random_forest 0.840753 0.868339 -0.027586 7.302905 2.815526e-13            False              True                   True
injury baseline_logistic|county_enriched|logistic_regression        baseline_logistic     county_enriched logistic_regression                    model_suite|county_selected|knn              model_suite     county_selected                 knn 0.867775 0.840753  0.027022 7.017464 2.259304e-12            False             False                  False
injury                       model_suite|county_selected|knn              model_suite     county_selected                 knn    model_suite|county_selected|logistic_regression              model_suite     county_selected logistic_regression 0.840753 0.867452 -0.026699 6.804370 1.014921e-11            False              True                   True
injury                       model_suite|county_selected|knn              model_suite     county_selected                 knn                    model_suite|county_selected|svm              model_suite     county_selected                 svm 0.840753 0.867386 -0.026633 6.781711 1.187606e-11            False              True                   True
injury    baseline_logistic|tornado_only|logistic_regression        baseline_logistic        tornado_only logistic_regression                    model_suite|county_selected|knn              model_suite     county_selected                 knn 0.866128 0.840753  0.025375 6.347037 2.195011e-10            False             False                  False
injury    baseline_logistic|tornado_only|logistic_regression        baseline_logistic        tornado_only logistic_regression         model_suite|county_selected|neural_network              model_suite     county_selected      neural_network 0.866128 0.870758 -0.004629 3.552201 3.820223e-04            False             False                  False
injury       model_suite|county_selected|logistic_regression              model_suite     county_selected logistic_regression         model_suite|county_selected|neural_network              model_suite     county_selected      neural_network 0.867452 0.870758 -0.003306 3.405224 6.610985e-04            False              True                   True
injury            model_suite|county_selected|neural_network              model_suite     county_selected      neural_network                    model_suite|county_selected|svm              model_suite     county_selected                 svm 0.870758 0.867386  0.003372 3.280679 1.035574e-03            False              True                   True
injury    baseline_logistic|tornado_only|logistic_regression        baseline_logistic        tornado_only logistic_regression                model_suite|county_selected|xgboost              model_suite     county_selected             xgboost 0.866128 0.871518 -0.005390 3.167196 1.539166e-03            False             False                  False
injury baseline_logistic|county_enriched|logistic_regression        baseline_logistic     county_enriched logistic_regression         model_suite|county_selected|neural_network              model_suite     county_selected      neural_network 0.867775 0.870758 -0.002983 3.005757 2.649202e-03            False             False                  False
injury       model_suite|county_selected|logistic_regression              model_suite     county_selected logistic_regression                model_suite|county_selected|xgboost              model_suite     county_selected             xgboost 0.867452 0.871518 -0.004066 2.727513 6.381374e-03            False              True                   True
injury                       model_suite|county_selected|svm              model_suite     county_selected                 svm                model_suite|county_selected|xgboost              model_suite     county_selected             xgboost 0.867386 0.871518 -0.004132 2.715544 6.616701e-03            False              True                   True
injury baseline_logistic|county_enriched|logistic_regression        baseline_logistic     county_enriched logistic_regression                model_suite|county_selected|xgboost              model_suite     county_selected             xgboost 0.867775 0.871518 -0.003743 2.554089 1.064659e-02            False             False                  False
injury             model_suite|county_selected|random_forest              model_suite     county_selected       random_forest                model_suite|county_selected|xgboost              model_suite     county_selected             xgboost 0.868339 0.871518 -0.003179 2.128849 3.326674e-02            False              True                   True
injury baseline_logistic|county_enriched|logistic_regression        baseline_logistic     county_enriched logistic_regression baseline_logistic|tornado_only|logistic_regression        baseline_logistic        tornado_only logistic_regression 0.867775 0.866128  0.001646 1.619471 1.053460e-01             True             False                   True
injury    baseline_logistic|tornado_only|logistic_regression        baseline_logistic        tornado_only logistic_regression    model_suite|county_selected|logistic_regression              model_suite     county_selected logistic_regression 0.866128 0.867452 -0.001324 1.457094 1.450904e-01             True             False                  False
injury    baseline_logistic|tornado_only|logistic_regression        baseline_logistic        tornado_only logistic_regression                    model_suite|county_selected|svm              model_suite     county_selected                 svm 0.866128 0.867386 -0.001258 1.380394 1.674654e-01            False             False                  False
injury            model_suite|county_selected|neural_network              model_suite     county_selected      neural_network          model_suite|county_selected|random_forest              model_suite     county_selected       random_forest 0.870758 0.868339  0.002419 1.231177 2.182568e-01            False              True                   True
injury    baseline_logistic|tornado_only|logistic_regression        baseline_logistic        tornado_only logistic_regression          model_suite|county_selected|random_forest              model_suite     county_selected       random_forest 0.866128 0.868339 -0.002211 0.943221 3.455677e-01            False             False                  False
injury baseline_logistic|county_enriched|logistic_regression        baseline_logistic     county_enriched logistic_regression                    model_suite|county_selected|svm              model_suite     county_selected                 svm 0.867775 0.867386  0.000388 0.736537 4.614038e-01            False             False                  False
injury baseline_logistic|county_enriched|logistic_regression        baseline_logistic     county_enriched logistic_regression    model_suite|county_selected|logistic_regression              model_suite     county_selected logistic_regression 0.867775 0.867452  0.000323 0.635661 5.249972e-01             True             False                  False
injury            model_suite|county_selected|neural_network              model_suite     county_selected      neural_network                model_suite|county_selected|xgboost              model_suite     county_selected             xgboost 0.870758 0.871518 -0.000760 0.570486 5.683478e-01            False              True                   True
injury             model_suite|county_selected|random_forest              model_suite     county_selected       random_forest                    model_suite|county_selected|svm              model_suite     county_selected                 svm 0.868339 0.867386  0.000953 0.420297 6.742686e-01            False              True                   True
injury       model_suite|county_selected|logistic_regression              model_suite     county_selected logistic_regression          model_suite|county_selected|random_forest              model_suite     county_selected       random_forest 0.867452 0.868339 -0.000887 0.398317 6.903962e-01            False              True                   True
injury       model_suite|county_selected|logistic_regression              model_suite     county_selected logistic_regression                    model_suite|county_selected|svm              model_suite     county_selected                 svm 0.867452 0.867386  0.000066 0.291207 7.708929e-01            False              True                   True
injury baseline_logistic|county_enriched|logistic_regression        baseline_logistic     county_enriched logistic_regression          model_suite|county_selected|random_forest              model_suite     county_selected       random_forest 0.867775 0.868339 -0.000565 0.256187 7.978067e-01            False             False                  False
```

### Injury Same-Model Feature-Set DeLong

```text
  task                                            model_1_id model_1_experiment_group model_1_feature_set        model_1_name                                         model_2_id model_2_experiment_group model_2_feature_set        model_2_name    auc_1    auc_2  auc_diff        z  p_value  same_model_name  same_feature_set  same_experiment_group
injury baseline_logistic|county_enriched|logistic_regression        baseline_logistic     county_enriched logistic_regression baseline_logistic|tornado_only|logistic_regression        baseline_logistic        tornado_only logistic_regression 0.867775 0.866128  0.001646 1.619471 0.105346             True             False                   True
injury    baseline_logistic|tornado_only|logistic_regression        baseline_logistic        tornado_only logistic_regression    model_suite|county_selected|logistic_regression              model_suite     county_selected logistic_regression 0.866128 0.867452 -0.001324 1.457094 0.145090             True             False                  False
injury baseline_logistic|county_enriched|logistic_regression        baseline_logistic     county_enriched logistic_regression    model_suite|county_selected|logistic_regression              model_suite     county_selected logistic_regression 0.867775 0.867452  0.000323 0.635661 0.524997             True             False                  False
```

### Fatality All-Pairs DeLong

```text
    task                                            model_1_id model_1_experiment_group model_1_feature_set        model_1_name                                         model_2_id model_2_experiment_group model_2_feature_set        model_2_name    auc_1    auc_2  auc_diff        z      p_value  same_model_name  same_feature_set  same_experiment_group
fatality baseline_logistic|county_enriched|logistic_regression        baseline_logistic     county_enriched logistic_regression                    model_suite|county_selected|knn              model_suite     county_selected                 knn 0.926940 0.872990  0.053950 5.477400 4.316213e-08            False             False                  False
fatality                       model_suite|county_selected|knn              model_suite     county_selected                 knn    model_suite|county_selected|logistic_regression              model_suite     county_selected logistic_regression 0.872990 0.926196 -0.053206 5.430266 5.627018e-08            False              True                   True
fatality                       model_suite|county_selected|knn              model_suite     county_selected                 knn                model_suite|county_selected|xgboost              model_suite     county_selected             xgboost 0.872990 0.923363 -0.050373 5.416589 6.074667e-08            False              True                   True
fatality    baseline_logistic|tornado_only|logistic_regression        baseline_logistic        tornado_only logistic_regression                    model_suite|county_selected|knn              model_suite     county_selected                 knn 0.925191 0.872990  0.052201 5.403191 6.546554e-08            False             False                  False
fatality                       model_suite|county_selected|knn              model_suite     county_selected                 knn                    model_suite|county_selected|svm              model_suite     county_selected                 svm 0.872990 0.925629 -0.052639 5.373159 7.736890e-08            False              True                   True
fatality                       model_suite|county_selected|knn              model_suite     county_selected                 knn         model_suite|county_selected|neural_network              model_suite     county_selected      neural_network 0.872990 0.922070 -0.049080 5.304023 1.132783e-07            False              True                   True
fatality                       model_suite|county_selected|knn              model_suite     county_selected                 knn          model_suite|county_selected|random_forest              model_suite     county_selected       random_forest 0.872990 0.910602 -0.037612 4.362107 1.288156e-05            False              True                   True
fatality baseline_logistic|county_enriched|logistic_regression        baseline_logistic     county_enriched logistic_regression          model_suite|county_selected|random_forest              model_suite     county_selected       random_forest 0.926940 0.910602  0.016339 3.339648 8.388457e-04            False             False                  False
fatality       model_suite|county_selected|logistic_regression              model_suite     county_selected logistic_regression          model_suite|county_selected|random_forest              model_suite     county_selected       random_forest 0.926196 0.910602  0.015594 3.181917 1.463036e-03            False              True                   True
fatality    baseline_logistic|tornado_only|logistic_regression        baseline_logistic        tornado_only logistic_regression          model_suite|county_selected|random_forest              model_suite     county_selected       random_forest 0.925191 0.910602  0.014589 3.111525 1.861234e-03            False             False                  False
fatality             model_suite|county_selected|random_forest              model_suite     county_selected       random_forest                    model_suite|county_selected|svm              model_suite     county_selected                 svm 0.910602 0.925629 -0.015027 3.037947 2.381959e-03            False              True                   True
fatality             model_suite|county_selected|random_forest              model_suite     county_selected       random_forest                model_suite|county_selected|xgboost              model_suite     county_selected             xgboost 0.910602 0.923363 -0.012761 2.775288 5.515278e-03            False              True                   True
fatality            model_suite|county_selected|neural_network              model_suite     county_selected      neural_network          model_suite|county_selected|random_forest              model_suite     county_selected       random_forest 0.922070 0.910602  0.011468 2.304870 2.117384e-02            False              True                   True
fatality       model_suite|county_selected|logistic_regression              model_suite     county_selected logistic_regression                    model_suite|county_selected|svm              model_suite     county_selected                 svm 0.926196 0.925629  0.000567 2.134633 3.279098e-02            False              True                   True
fatality baseline_logistic|county_enriched|logistic_regression        baseline_logistic     county_enriched logistic_regression                    model_suite|county_selected|svm              model_suite     county_selected                 svm 0.926940 0.925629  0.001311 1.980852 4.760789e-02            False             False                  False
fatality baseline_logistic|county_enriched|logistic_regression        baseline_logistic     county_enriched logistic_regression         model_suite|county_selected|neural_network              model_suite     county_selected      neural_network 0.926940 0.922070  0.004870 1.786083 7.408574e-02            False             False                  False
fatality       model_suite|county_selected|logistic_regression              model_suite     county_selected logistic_regression         model_suite|county_selected|neural_network              model_suite     county_selected      neural_network 0.926196 0.922070  0.004126 1.568077 1.168632e-01            False              True                   True
fatality            model_suite|county_selected|neural_network              model_suite     county_selected      neural_network                    model_suite|county_selected|svm              model_suite     county_selected                 svm 0.922070 0.925629 -0.003559 1.352823 1.761122e-01            False              True                   True
fatality baseline_logistic|county_enriched|logistic_regression        baseline_logistic     county_enriched logistic_regression                model_suite|county_selected|xgboost              model_suite     county_selected             xgboost 0.926940 0.923363  0.003577 1.252335 2.104477e-01            False             False                  False
fatality baseline_logistic|county_enriched|logistic_regression        baseline_logistic     county_enriched logistic_regression    model_suite|county_selected|logistic_regression              model_suite     county_selected logistic_regression 0.926940 0.926196  0.000744 1.214525 2.245474e-01             True             False                  False
fatality    baseline_logistic|tornado_only|logistic_regression        baseline_logistic        tornado_only logistic_regression         model_suite|county_selected|neural_network              model_suite     county_selected      neural_network 0.925191 0.922070  0.003121 0.994184 3.201333e-01            False             False                  False
fatality baseline_logistic|county_enriched|logistic_regression        baseline_logistic     county_enriched logistic_regression baseline_logistic|tornado_only|logistic_regression        baseline_logistic        tornado_only logistic_regression 0.926940 0.925191  0.001749 0.976000 3.290645e-01             True             False                   True
fatality       model_suite|county_selected|logistic_regression              model_suite     county_selected logistic_regression                model_suite|county_selected|xgboost              model_suite     county_selected             xgboost 0.926196 0.923363  0.002833 0.963191 3.354516e-01            False              True                   True
fatality                       model_suite|county_selected|svm              model_suite     county_selected                 svm                model_suite|county_selected|xgboost              model_suite     county_selected             xgboost 0.925629 0.923363  0.002266 0.767414 4.428351e-01            False              True                   True
fatality    baseline_logistic|tornado_only|logistic_regression        baseline_logistic        tornado_only logistic_regression                model_suite|county_selected|xgboost              model_suite     county_selected             xgboost 0.925191 0.923363  0.001828 0.606496 5.441851e-01            False             False                  False
fatality    baseline_logistic|tornado_only|logistic_regression        baseline_logistic        tornado_only logistic_regression    model_suite|county_selected|logistic_regression              model_suite     county_selected logistic_regression 0.925191 0.926196 -0.001005 0.510181 6.099244e-01             True             False                  False
fatality            model_suite|county_selected|neural_network              model_suite     county_selected      neural_network                model_suite|county_selected|xgboost              model_suite     county_selected             xgboost 0.922070 0.923363 -0.001293 0.499291 6.175747e-01            False              True                   True
fatality    baseline_logistic|tornado_only|logistic_regression        baseline_logistic        tornado_only logistic_regression                    model_suite|county_selected|svm              model_suite     county_selected                 svm 0.925191 0.925629 -0.000438 0.224567 8.223160e-01            False             False                  False
```

### Fatality Same-Model Feature-Set DeLong

```text
    task                                            model_1_id model_1_experiment_group model_1_feature_set        model_1_name                                         model_2_id model_2_experiment_group model_2_feature_set        model_2_name    auc_1    auc_2  auc_diff        z  p_value  same_model_name  same_feature_set  same_experiment_group
fatality baseline_logistic|county_enriched|logistic_regression        baseline_logistic     county_enriched logistic_regression    model_suite|county_selected|logistic_regression              model_suite     county_selected logistic_regression 0.926940 0.926196  0.000744 1.214525 0.224547             True             False                  False
fatality baseline_logistic|county_enriched|logistic_regression        baseline_logistic     county_enriched logistic_regression baseline_logistic|tornado_only|logistic_regression        baseline_logistic        tornado_only logistic_regression 0.926940 0.925191  0.001749 0.976000 0.329064             True             False                   True
fatality    baseline_logistic|tornado_only|logistic_regression        baseline_logistic        tornado_only logistic_regression    model_suite|county_selected|logistic_regression              model_suite     county_selected logistic_regression 0.925191 0.926196 -0.001005 0.510181 0.609924             True             False                  False
```

### Significant DeLong ROC Plot Manifest

```text
    task                                            model_1_id                                      model_2_id    auc_1    auc_2  auc_diff        z      p_value                          plot_file
fatality baseline_logistic|county_enriched|logistic_regression                 model_suite|county_selected|knn 0.926940 0.872990  0.053950 5.477400 4.316213e-08 fatality_delong_significant_01.png
fatality                       model_suite|county_selected|knn model_suite|county_selected|logistic_regression 0.872990 0.926196 -0.053206 5.430266 5.627018e-08 fatality_delong_significant_02.png
fatality                       model_suite|county_selected|knn             model_suite|county_selected|xgboost 0.872990 0.923363 -0.050373 5.416589 6.074667e-08 fatality_delong_significant_03.png
fatality    baseline_logistic|tornado_only|logistic_regression                 model_suite|county_selected|knn 0.925191 0.872990  0.052201 5.403191 6.546554e-08 fatality_delong_significant_04.png
fatality                       model_suite|county_selected|knn                 model_suite|county_selected|svm 0.872990 0.925629 -0.052639 5.373159 7.736890e-08 fatality_delong_significant_05.png
fatality                       model_suite|county_selected|knn      model_suite|county_selected|neural_network 0.872990 0.922070 -0.049080 5.304023 1.132783e-07 fatality_delong_significant_06.png
fatality                       model_suite|county_selected|knn       model_suite|county_selected|random_forest 0.872990 0.910602 -0.037612 4.362107 1.288156e-05 fatality_delong_significant_07.png
fatality baseline_logistic|county_enriched|logistic_regression       model_suite|county_selected|random_forest 0.926940 0.910602  0.016339 3.339648 8.388457e-04 fatality_delong_significant_08.png
fatality       model_suite|county_selected|logistic_regression       model_suite|county_selected|random_forest 0.926196 0.910602  0.015594 3.181917 1.463036e-03 fatality_delong_significant_09.png
fatality    baseline_logistic|tornado_only|logistic_regression       model_suite|county_selected|random_forest 0.925191 0.910602  0.014589 3.111525 1.861234e-03 fatality_delong_significant_10.png
fatality             model_suite|county_selected|random_forest                 model_suite|county_selected|svm 0.910602 0.925629 -0.015027 3.037947 2.381959e-03 fatality_delong_significant_11.png
fatality             model_suite|county_selected|random_forest             model_suite|county_selected|xgboost 0.910602 0.923363 -0.012761 2.775288 5.515278e-03 fatality_delong_significant_12.png
fatality            model_suite|county_selected|neural_network       model_suite|county_selected|random_forest 0.922070 0.910602  0.011468 2.304870 2.117384e-02 fatality_delong_significant_13.png
fatality       model_suite|county_selected|logistic_regression                 model_suite|county_selected|svm 0.926196 0.925629  0.000567 2.134633 3.279098e-02 fatality_delong_significant_14.png
fatality baseline_logistic|county_enriched|logistic_regression                 model_suite|county_selected|svm 0.926940 0.925629  0.001311 1.980852 4.760789e-02 fatality_delong_significant_15.png
  injury                       model_suite|county_selected|knn             model_suite|county_selected|xgboost 0.840753 0.871518 -0.030765 8.006955 1.110223e-15   injury_delong_significant_16.png
  injury                       model_suite|county_selected|knn      model_suite|county_selected|neural_network 0.840753 0.870758 -0.030005 7.851934 3.996803e-15   injury_delong_significant_17.png
  injury                       model_suite|county_selected|knn       model_suite|county_selected|random_forest 0.840753 0.868339 -0.027586 7.302905 2.815526e-13   injury_delong_significant_18.png
  injury baseline_logistic|county_enriched|logistic_regression                 model_suite|county_selected|knn 0.867775 0.840753  0.027022 7.017464 2.259304e-12   injury_delong_significant_19.png
  injury                       model_suite|county_selected|knn model_suite|county_selected|logistic_regression 0.840753 0.867452 -0.026699 6.804370 1.014921e-11   injury_delong_significant_20.png
  injury                       model_suite|county_selected|knn                 model_suite|county_selected|svm 0.840753 0.867386 -0.026633 6.781711 1.187606e-11   injury_delong_significant_21.png
  injury    baseline_logistic|tornado_only|logistic_regression                 model_suite|county_selected|knn 0.866128 0.840753  0.025375 6.347037 2.195011e-10   injury_delong_significant_22.png
  injury    baseline_logistic|tornado_only|logistic_regression      model_suite|county_selected|neural_network 0.866128 0.870758 -0.004629 3.552201 3.820223e-04   injury_delong_significant_23.png
  injury       model_suite|county_selected|logistic_regression      model_suite|county_selected|neural_network 0.867452 0.870758 -0.003306 3.405224 6.610985e-04   injury_delong_significant_24.png
  injury            model_suite|county_selected|neural_network                 model_suite|county_selected|svm 0.870758 0.867386  0.003372 3.280679 1.035574e-03   injury_delong_significant_25.png
  injury    baseline_logistic|tornado_only|logistic_regression             model_suite|county_selected|xgboost 0.866128 0.871518 -0.005390 3.167196 1.539166e-03   injury_delong_significant_26.png
  injury baseline_logistic|county_enriched|logistic_regression      model_suite|county_selected|neural_network 0.867775 0.870758 -0.002983 3.005757 2.649202e-03   injury_delong_significant_27.png
  injury       model_suite|county_selected|logistic_regression             model_suite|county_selected|xgboost 0.867452 0.871518 -0.004066 2.727513 6.381374e-03   injury_delong_significant_28.png
  injury                       model_suite|county_selected|svm             model_suite|county_selected|xgboost 0.867386 0.871518 -0.004132 2.715544 6.616701e-03   injury_delong_significant_29.png
  injury baseline_logistic|county_enriched|logistic_regression             model_suite|county_selected|xgboost 0.867775 0.871518 -0.003743 2.554089 1.064659e-02   injury_delong_significant_30.png
  injury             model_suite|county_selected|random_forest             model_suite|county_selected|xgboost 0.868339 0.871518 -0.003179 2.128849 3.326674e-02   injury_delong_significant_31.png
```

## SHAP Analysis

SHAP was computed for the best validation-AUPRC model in each task. In this run, the best model was XGBoost for both injury and fatality.

### Injury SHAP Global Summary

```text
  task best_model_name                             feature  mean_abs_shap  rank
injury         xgboost                        num__mag_num       1.253860     1
injury         xgboost    num__path_unemployment_rate_wavg       0.303625     2
injury         xgboost num__start_county_mobile_home_share       0.245958     3
injury         xgboost   num__path_population_density_wavg       0.135246     4
injury         xgboost                cat__region_Southern       0.106798     5
injury         xgboost                  cat__season_Spring       0.105687     6
injury         xgboost                    num__mag_missing       0.088600     7
injury         xgboost             cat__region_High Plains       0.068568     8
injury         xgboost                       num__len_zero       0.052919     9
injury         xgboost                 cat__region_Western       0.050487    10
injury         xgboost   num__start_county_disability_rate       0.050114    11
injury         xgboost                            num__len       0.026257    12
injury         xgboost          num__path_overlap_area_km2       0.022338    13
injury         xgboost                  cat__season_Summer       0.021407    14
injury         xgboost  num__start_county_age_65plus_share       0.016865    15
injury         xgboost           num__path_est_exposed_pop       0.015144    16
injury         xgboost                    num__end_missing       0.000720    17
injury         xgboost                  cat__season_Winter       0.000194    18
injury         xgboost               cat__region_Northeast       0.000069    19
injury         xgboost               cat__region_Southeast       0.000000    20
```

![injury SHAP beeswarm](C:/Users/singc/OneDrive/Documents/bruh code/4641/Project/final out/plots/injury_best_model_shap_beeswarm.png)

![injury SHAP bar](C:/Users/singc/OneDrive/Documents/bruh code/4641/Project/final out/plots/injury_best_model_shap_bar.png)

![injury SHAP waterfall](C:/Users/singc/OneDrive/Documents/bruh code/4641/Project/final out/plots/injury_best_model_shap_waterfall_highest_risk.png)

### Injury SHAP Local Contributions

```text
  task                case  predicted_score  true_label  rank                             feature  shap_value  feature_value
injury highest_risk_sample         0.987308           1     1                        num__mag_num    3.102072       3.596358
injury highest_risk_sample         0.987308           1     2 num__start_county_mobile_home_share    0.789693       2.722465
injury highest_risk_sample         0.987308           1     3                  cat__season_Spring    0.288994       0.137670
injury highest_risk_sample         0.987308           1     4                cat__region_Southern    0.131784       0.636048
injury highest_risk_sample         0.987308           1     5             cat__region_High Plains    0.102970       1.584619
injury highest_risk_sample         0.987308           1     6   num__path_population_density_wavg   -0.060210      -0.796383
injury highest_risk_sample         0.987308           1     7    num__path_unemployment_rate_wavg   -0.052135      -0.288159
injury highest_risk_sample         0.987308           1     8                 cat__region_Western    0.041908       0.156243
injury highest_risk_sample         0.987308           1     9                    num__mag_missing    0.030263       0.000000
injury highest_risk_sample         0.987308           1    10   num__start_county_disability_rate   -0.024539       0.000000
```

### Fatality SHAP Global Summary

```text
    task best_model_name                             feature  mean_abs_shap  rank
fatality         xgboost                        num__mag_num       1.968890     1
fatality         xgboost    num__end_county_age_65plus_share       0.315339     2
fatality         xgboost                 cat__region_Western       0.304980     3
fatality         xgboost                    num__mag_missing       0.132419     4
fatality         xgboost     num__end_county_disability_rate       0.113264     5
fatality         xgboost    num__path_age_under18_share_wavg       0.105780     6
fatality         xgboost                  cat__season_Winter       0.085716     7
fatality         xgboost               cat__region_Northeast       0.073024     8
fatality         xgboost   num__end_county_unemployment_rate       0.064359     9
fatality         xgboost                  cat__season_Spring       0.062283    10
fatality         xgboost     num__path_age_65plus_share_wavg       0.055111    11
fatality         xgboost      num__path_disability_rate_wavg       0.040999    12
fatality         xgboost                            num__wid       0.039772    13
fatality         xgboost                            num__len       0.039066    14
fatality         xgboost num__start_county_unemployment_rate       0.035764    15
fatality         xgboost                 cat__region_Midwest       0.034375    16
fatality         xgboost           num__path_est_exposed_pop       0.030394    17
fatality         xgboost   num__start_county_disability_rate       0.013646    18
fatality         xgboost             cat__region_High Plains       0.000938    19
fatality         xgboost num__start_county_mobile_home_share       0.000000    20
```

![fatality SHAP beeswarm](C:/Users/singc/OneDrive/Documents/bruh code/4641/Project/final out/plots/fatality_best_model_shap_beeswarm.png)

![fatality SHAP bar](C:/Users/singc/OneDrive/Documents/bruh code/4641/Project/final out/plots/fatality_best_model_shap_bar.png)

![fatality SHAP waterfall](C:/Users/singc/OneDrive/Documents/bruh code/4641/Project/final out/plots/fatality_best_model_shap_waterfall_highest_risk.png)

### Fatality SHAP Local Contributions

```text
    task                case  predicted_score  true_label  rank                             feature  shap_value  feature_value
fatality highest_risk_sample         0.993618           1     1                        num__mag_num    3.562362       3.602889
fatality highest_risk_sample         0.993618           1     2                 cat__region_Western    0.762632       2.191108
fatality highest_risk_sample         0.993618           1     3    num__end_county_age_65plus_share    0.756013       8.932126
fatality highest_risk_sample         0.993618           1     4      num__path_disability_rate_wavg   -0.042321       2.388573
fatality highest_risk_sample         0.993618           1     5                    num__mag_missing    0.034571       0.000000
fatality highest_risk_sample         0.993618           1     6     num__end_county_disability_rate    0.029169      -0.108230
fatality highest_risk_sample         0.993618           1     7   num__end_county_unemployment_rate    0.028703       0.839070
fatality highest_risk_sample         0.993618           1     8                  cat__season_Spring   -0.026669       1.131925
fatality highest_risk_sample         0.993618           1     9 num__start_county_unemployment_rate   -0.025660       0.000000
fatality highest_risk_sample         0.993618           1    10     num__path_age_65plus_share_wavg   -0.021153       1.000000
```

## Output Inventory

- `master_report.md`: this report
- `tables/full_model_results_detailed.csv`: every saved metric plus confusion counts and best params
- `tables/pairwise_delong_all_models.csv`: exhaustive AUROC DeLong comparisons
- `tables/pairwise_delong_same_model_feature_sets.csv`: same-model feature-set comparisons
- `tables/significant_delong_plot_manifest.csv`: file map for significant DeLong ROC plots
- `tables/dataset_summary.csv`: split sizes and class rates
- `tables/shap_global_summary.csv`: ranked mean absolute SHAP values for the best model in each task
- `tables/shap_local_contributions.csv`: top local SHAP contributions for the highest-risk sample in each task
- `plots/*.png`: SHAP beeswarm, bar, and waterfall plots