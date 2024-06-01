#!/bin/bash
# 5
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
python run_fed_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-even missing_scenario_name=mnar2-homog fed_strategy=fedprox round_id=1
python run_fed_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-even missing_scenario_name=mnar2-homog fed_strategy=fedprox round_id=2
python run_fed_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-even missing_scenario_name=mnar2-homog fed_strategy=fedprox round_id=3