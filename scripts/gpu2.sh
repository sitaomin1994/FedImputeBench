#!/bin/bash
# fedavg
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
python run_fed_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-even missing_scenario_name=mcar fed_strategy=central round_id=0
python run_fed_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-even missing_scenario_name=mcar fed_strategy=central round_id=4
python run_fed_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-even missing_scenario_name=mar-homog fed_strategy=central round_id=0
python run_fed_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-even missing_scenario_name=mar-homog fed_strategy=central round_id=2
python run_fed_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=niid-t1 missing_scenario_name=mnar2-heter fed_strategy=central round_id=3
python run_fed_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=niid-t1 missing_scenario_name=mnar2-heter fed_strategy=central round_id=4
python run_fed_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=niid-t2 missing_scenario_name=mcar fed_strategy=central round_id=0
python run_fed_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=niid-t2 missing_scenario_name=mcar fed_strategy=central round_id=4
python run_fed_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=niid-t2 missing_scenario_name=mar-homog fed_strategy=central round_id=0
python run_fed_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=niid-t2 missing_scenario_name=mar-homog fed_strategy=central round_id=1