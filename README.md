
## FedImputeBench: A Benchmarking Analysis for Federated Imputation

This is the repo for reproducing a simple pilot study (benchmarking analysis) of federated imputation replies on [FedImpute](https://github.com/idsla/FedImpute). Note that **FedImputeBench** is developed using the initial version of FedImpute; the API might be slightly different. Refer to [FedImpute Documentation](https://idsla.github.io/FedImpute/) for the latest API for using FedImpute.

### Pre-setup
```shell
conda env create -f environment.yml
conda activate fed_imp
python setup_project.py
```

### Configuration Management

We rely on Hydra-core to manage the configuration of experiments. All configuration files are located in `\config` with different folders representing different configurations:

- **Data Partition:** `\config\data_partition\`
- **Missing Scenario:** `\config\missing_scenario\`
- **Imputer:** `\config\imputer\`
- **Federated Strategy:** `\config\fed_strategy\`
- **Hyperparameters:** `\config\hyper_params\`

### Dataset and Preprocessing

Datasets after preprocessing and configuration are in `/data`. Preprocessing scripts for datasets are in `/notebooks/data_processing.ipynb`

### Run Federated Missing Data Simulation (Example)

```python
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
# single run
python run_scenario_generation.py dataset_name=codrna data_partition=iid-even missing_scenario=mcar

# multiple runs
python run_scenario_generation.py -m dataset_name=codrna data_partition=iid-even,iid-uneven
missing_scenario=mcar,mar-homog,mar-homo,mar-heter
```

### Run Federated Imputation Algorithms (Example)

```python
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
# single run
python run_fed_imp_scenario.py dataset_name=codrna data_partition_name=iid-even missing_scenario_name=mcar
imputer=gain fed_strategy=fedavg round_id=0 experiment.log_to_file=True

# multiple runs
python run_fed_imp_scenario.py -m dataset_name=codrna data_partition_name=iid-even,iid-uneven missing_scenario_name=mcar,mar-heter,mar-homog
imputer=gain fed_strategy=fedavg round_id=0,1,2,3,4,5 experiment.log_to_file=True
```

### Run Evaluation (Example)

```python
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# single run
python run_evaluation.py dataset_name=codrna imputer_name=gain data_partition_name=iid-even 
missing_scenario_name=mcar fed_strategy_name=fedavg round_idx=0 eval_params.model=nn  
log_to_file=False log_level=SUCCESS experiment_name=fed_imp_pc  # experiment name should match the one used in run_fed_imp_scenario.py see /config/imp_config_p.yaml - experiment_name field

# multiple runs
python run_evaluation.py -m dataset_name=codrna imputer_name=gain,miwae data_partition_name=iid-even,iid-uneven 
missing_scenario_name=mcar,mar-homog fed_strategy_name=fedavg,fedprox round_idx=0,1,2,3,4,5 eval_params.model=nn  
log_to_file=False log_level=SUCCESS experiment_name=fed_imp_pc
```

### Batch Experiments Scripts

Scripts for running batch experiments using multiple processing or in HPC are in `/scripts/`

### Plots in Paper

Scripts for generating plots of in the paper are in `/notebooks/result*.ipynb`

