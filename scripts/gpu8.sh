#!/bin/bash
# 7
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

python run_fed_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-uneven missing_scenario_name=mnar2-heter fed_strategy=central round_id=2
python run_fed_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-uneven missing_scenario_name=mnar2-heter fed_strategy=central round_id=4