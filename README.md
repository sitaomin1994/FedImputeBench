
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

### Run Federated Missing Data Simulation

```
```

### Run Federated Imputation Algorithms

### Run Evaluation



