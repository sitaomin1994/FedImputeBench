## Documentation

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