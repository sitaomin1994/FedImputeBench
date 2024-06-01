#!/bin/bash
# 17
#./scripts/run_scenario_gpu.sh hhip iid-even mcar,mnar2-heter,mnar2-homog miwae central 1
#./scripts/run_scenario_gpu.sh hhip iid-uneven mar-heter,mar-homog,mnar2-heter,mnar2-homog miwae central 1
#./scripts/run_scenario_gpu.sh hhip niid-t1,niid-t2 mcar,mar-homog,mar-heter,mnar2-heter,mnar2-homog miwae central 1
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# local
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-even missing_scenario_name=mar-homog fed_strategy=local round_id=4
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-uneven missing_scenario_name=mar-homog fed_strategy=local round_id=1
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-uneven missing_scenario_name=mnar2-heter fed_strategy=local round_id=3
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=niid-t1 missing_scenario_name=mcar fed_strategy=local round_id=4
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=niid-t1 missing_scenario_name=mnar2-heter fed_strategy=local round_id=3

# fedavg
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-even missing_scenario_name=mar-homog fed_strategy=fedavg round_id=4
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-uneven missing_scenario_name=mcar fed_strategy=fedavg round_id=1
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-uneven missing_scenario_name=mar-homog fed_strategy=fedavg round_id=2
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-uneven missing_scenario_name=mnar2-homog fed_strategy=fedavg round_id=4
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-uneven missing_scenario_name=mnar2-heter fed_strategy=fedavg round_id=0
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=niid-t2 missing_scenario_name=mnar2-homog fed_strategy=fedavg round_id=1

# fedprox
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-even missing_scenario_name=mnar2-heter fed_strategy=fedprox round_id=0
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-uneven missing_scenario_name=mcar fed_strategy=fedprox round_id=0
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-uneven missing_scenario_name=mcar fed_strategy=fedprox round_id=1
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-uneven missing_scenario_name=mnar2-homog fed_strategy=fedprox round_id=0
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=iid-uneven missing_scenario_name=mnar2-homog fed_strategy=fedprox round_id=4
python run_imp_scenario.py dataset_name=hhip imputer=miwae data_partition_name=niid-t1 missing_scenario_name=mar-homog fed_strategy=fedprox round_id=2



