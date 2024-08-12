#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=":4096:8"

#python run_evaluation.py dataset_name=hhip hydra.launcher.n_jobs=8 \
#data_partition_name=iid-even missing_scenario_name=mcar imputer_name=miwae fed_strategy_name=local,fedavg_ft round_idx=0,1,2,3,4 eval_params.model=nn experiment_name=fed_imp_pc2
#python run_evaluation.py dataset_name=hhip hydra.launcher.n_jobs=8 \
#data_partition_name=iid-even missing_scenario_name=mar-homog imputer_name=miwae fed_strategy_name=local,fedavg,fedavg_ft round_idx=0,1,2,3,4 eval_params.model=nn experiment_name=fed_imp_pc2
#python run_evaluation.py dataset_name=hhip hydra.launcher.n_jobs=8 \
#data_partition_name=iid-even missing_scenario_name=mnar2-homog imputer_name=miwae fed_strategy_name=fedavg_ft,fedprox round_idx=0,1,2,3,4 eval_params.model=nn experiment_name=fed_imp_pc2
#python run_evaluation.py dataset_name=hhip hydra.launcher.n_jobs=8 \
#data_partition_name=iid-uneven missing_scenario_name=mnar2-homog imputer_name=miwae fed_strategy_name=fedavg,fedprox,fedavg_ft round_idx=0,1,2,3,4 eval_params.model=nn experiment_name=fed_imp_pc2
#python run_evaluation.py dataset_name=hhip hydra.launcher.n_jobs=8 \
#data_partition_name=iid-uneven missing_scenario_name=mnar2-heter imputer_name=miwae fed_strategy_name=local,fedavg,fedavg_ft round_idx=0,1,2,3,4 eval_params.model=nn experiment_name=fed_imp_pc2
#python run_evaluation.py dataset_name=hhip hydra.launcher.n_jobs=8 \
#data_partition_name=niid-t1 missing_scenario_name=mar-homog imputer_name=miwae fed_strategy_name=fedavg_ft,fedprox round_idx=0,1,2,3,4 eval_params.model=nn experiment_name=fed_imp_pc2
#python run_evaluation.py dataset_name=hhip hydra.launcher.n_jobs=8 \
#data_partition_name=niid-t1 missing_scenario_name=mar-heter imputer_name=miwae fed_strategy_name=fedavg_ft,fedprox round_idx=0,1,2,3,4 eval_params.model=nn experiment_name=fed_imp_pc2
#python run_evaluation.py dataset_name=hhip hydra.launcher.n_jobs=8 \
#data_partition_name=niid-t1 missing_scenario_name=mnar2-heter imputer_name=miwae fed_strategy_name=local,fedavg_ft round_idx=0,1,2,3,4 eval_params.model=nn experiment_name=fed_imp_pc2
#python run_evaluation.py dataset_name=hhip hydra.launcher.n_jobs=8 \
#data_partition_name=niid-t2 missing_scenario_name=mnar2-homog imputer_name=miwae fed_strategy_name=local,fedavg,fedavg_ft,fedprox round_idx=0,1,2,3,4 eval_params.model=nn experiment_name=fed_imp_pc2
#python run_evaluation.py dataset_name=hhip hydra.launcher.n_jobs=8 \
#data_partition_name=niid-t2 missing_scenario_name=mar-heter imputer_name=miwae fed_strategy_name=local,fedavg round_idx=0,1,2,3,4 eval_params.model=nn experiment_name=fed_imp_pc2

python run_evaluation.py -m dataset_name=hhip hydra.launcher.n_jobs=3 \
data_partition_name=iid-even missing_scenario_name=mnar2-heter imputer_name=miwae fed_strategy_name=local round_idx=0,1,2 eval_params.model=nn experiment_name=fed_imp_pc2
python run_evaluation.py -m dataset_name=hhip hydra.launcher.n_jobs=5 \
data_partition_name=iid-even missing_scenario_name=mnar2-heter imputer_name=miwae fed_strategy_name=fedprox round_idx=0,1,2,3,4 eval_params.model=nn experiment_name=fed_imp_pc2
python run_evaluation.py -m dataset_name=hhip hydra.launcher.n_jobs=5 \
data_partition_name=iid-uneven missing_scenario_name=mcar imputer_name=miwae fed_strategy_name=fedavg,fedprox round_idx=0,1,2,3,4 eval_params.model=nn experiment_name=fed_imp_pc2
python run_evaluation.py -m dataset_name=hhip hydra.launcher.n_jobs=4 \
data_partition_name=iid-uneven missing_scenario_name=mcar imputer_name=miwae fed_strategy_name=fedavg_ft round_idx=0,1,2,3 eval_params.model=nn experiment_name=fed_imp_pc2
python run_evaluation.py -m dataset_name=hhip hydra.launcher.n_jobs=1 \
data_partition_name=iid-uneven missing_scenario_name=mar-homog imputer_name=miwae fed_strategy_name=local round_idx=1 eval_params.model=nn experiment_name=fed_imp_pc2
python run_evaluation.py -m dataset_name=hhip hydra.launcher.n_jobs=1 \
data_partition_name=iid-uneven missing_scenario_name=mar-homog imputer_name=miwae fed_strategy_name=fedavg round_idx=2 eval_params.model=nn experiment_name=fed_imp_pc2
python run_evaluation.py -m dataset_name=hhip hydra.launcher.n_jobs=5 \
data_partition_name=iid-uneven missing_scenario_name=mar-heter imputer_name=miwae fed_strategy_name=fedavg_ft round_idx=0,1,2,3,4 eval_params.model=nn experiment_name=fed_imp_pc2
python run_evaluation.py -m dataset_name=hhip hydra.launcher.n_jobs=5 \
data_partition_name=niid-t1 missing_scenario_name=mcar imputer_name=miwae fed_strategy_name=local,fedavg_ft round_idx=0,1,2,3,4 eval_params.model=nn experiment_name=fed_imp_pc2
python run_evaluation.py -m dataset_name=hhip hydra.launcher.n_jobs=5 \
data_partition_name=niid-t1 missing_scenario_name=mnar2-homog imputer_name=miwae fed_strategy_name=fedavg_ft round_idx=0,1,2,3,4 eval_params.model=nn experiment_name=fed_imp_pc2
python run_evaluation.py -m dataset_name=hhip hydra.launcher.n_jobs=5 \
data_partition_name=niid-t2 missing_scenario_name=mcar imputer_name=miwae fed_strategy_name=fedavg_ft round_idx=0,1,2,3,4 eval_params.model=nn experiment_name=fed_imp_pc2
python run_evaluation.py -m dataset_name=hhip hydra.launcher.n_jobs=1 \
data_partition_name=niid-t2 missing_scenario_name=mar-heter imputer_name=miwae fed_strategy_name=fedavg_ft round_idx=0 eval_params.model=nn experiment_name=fed_imp_pc2

