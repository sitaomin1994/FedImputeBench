#!/bin/bash
# 5
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-uneven missing_scenario_name=mnar2-homog fed_strategy=fedavg_ft round_id=4
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-uneven missing_scenario_name=mar-heter fed_strategy=fedavg_ft round_id=0
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-uneven missing_scenario_name=mar-heter fed_strategy=fedavg_ft round_id=1
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-uneven missing_scenario_name=mar-heter fed_strategy=fedavg_ft round_id=2
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-uneven missing_scenario_name=mnar2-heter fed_strategy=fedavg_ft round_id=0
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-uneven missing_scenario_name=mnar2-heter fed_strategy=fedavg_ft round_id=1
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=niid-t1 missing_scenario_name=mar-homog fed_strategy=fedavg_ft round_id=0
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=niid-t1 missing_scenario_name=mar-homog fed_strategy=fedavg_ft round_id=2
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=niid-t1 missing_scenario_name=mar-homog fed_strategy=fedavg_ft round_id=3
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=niid-t1 missing_scenario_name=mnar2-homog fed_strategy=fedavg_ft round_id=1