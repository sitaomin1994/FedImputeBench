## Documentation

### Dataset Preparation

- Preferred data:
  - most of the features are numerical features (>50%) (only add missing to numerical features)
  - less than 200 features (prefer < 100 features)
  - prefer regression and multi-class classification (more classes better) problems
  - prefer large dataset (more than 10,000 samples, > 1000 is also ok)
  - high correlation between features (optional)
  - have feature that can make data naturally partitioned (e.g. hosital id) (optional)
  - dataset domain prefer: healthcare, finance, goverment, science etc.
  
- Data Preparation
  - Basic routines:
    - drop NA values if any
    - standardize numerical features (normalization and then min-max scaler)
    - change numerical features distribution to gaussian distribution (optional)
    - one-hot encoding for categorical features
    - sort columns following - numerical cols, binary cols, target
    - correlation analysis - determine which numerical feature is suitable for partition data (if target is not continous, we prefer a feature which is continous and highly correlated with all other features including target) - this will determine `split_col_idx` and `obs_col_idx` in `data_config` dict
    - sample data (optional) - if data is too large, sample data to make it smaller (20000) 
    - formalize a data config dictionary, the current structure is:
      ```python
      {
        'target': 'target_col_name', 
        'important_features_idx': [1,2,3], # list of highly correlated feature indices
        'features_idx': [0, 1, 2, 3, 4, 5], # feature index list
        'split_col_idx': [0],  # a list of feature which can be used to partition data exclude target (sorted in priortiy order)
        'obs_col_idx': [1, 4, 7],  # observation column index which is a list of columns where missing values are not present (e.g. in mar setting) (typically categorical columns)
        "num_cols": 10,  # the numerical columns 
        'task_type': 'classification',  # regression or classification
        'clf_type': 'binary-class', # binary-class, multi-class
        'data_type': 'tabular' # tabular, image, text
      }
      ```
  - Example data preprocessing function current verision is `src.modules.data_prep.data_prep_his.process_codrna()`

### Experiment Management

- ExperimentManager class
  - execute_experiment - `experiment.run()`, `experiment.save()`
- Experiment class
  - attribute: `file_backend`
  - methods
    - run - `self.run()` - `self.single_run()` `self.multiple_run()`
    - save - `self.save()`
    - utils - `self.merge_results()`


### Client and Server
- Client class
  - `fit_local_imp_model(params) -> model_params, fit_res`
  - `update_lcoal_imp_model(updated_model, params)`
  - `local_imputation(params)`
  - `initial_impute(imp_values, col_type)`
  - `calculate_data_utils(data_config)`
- Server class
  - `aggregate(local_models, fit_res) -> updated_models, agg_res`
  - `global_evaluation()`
- Workflow class
  - `run_fed_imp()` - `fed_imp_sequential(clients, server, evaluator, tracker)`, `fed_imp_parallel()`

### Imputer, Imp_workflow, Agg_strategy
- Imputer class
  - initialize model and params `initialize(params, seed)`
  - fetch params: `get_imp_model_params(params) -> model_params`
  - update params: `set_imp_model_params(updated_model, params)`
  - fit imp model: `fit(X, y, missing_mask, params) -> fit_res`
  - impute: `impute(X, y, missing_mask, params) -> X_imp`
- AggStrategy class
  - `aggregate(local_models, fit_res) -> updated_models, agg_res`