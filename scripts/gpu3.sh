#!/bin/bash
# 14
# fedprox
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
python run_fed_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=niid-t1 missing_scenario_name=mar-heter fed_strategy=fedavg_ft round_id=2
python run_fed_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=niid-t1 missing_scenario_name=mar-heter fed_strategy=fedavg_ft round_id=4
python run_fed_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=niid-t2 missing_scenario_name=mar-heter fed_strategy=fedavg_ft round_id=0