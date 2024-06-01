#!/bin/bash
# fedavg
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
python run_fed_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=niid-t1 missing_scenario_name=mcar fed_strategy=fedavg_ft round_id=0
python run_fed_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=niid-t1 missing_scenario_name=mcar fed_strategy=fedavg_ft round_id=1
python run_fed_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=niid-t1 missing_scenario_name=mcar fed_strategy=fedavg_ft round_id=3
