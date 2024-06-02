#!/bin/bash

# miwae
python run_evaluation.py --multirun hydra.launcher.n_jobs=-1 dataset_name=hhip \
data_partition_name=niid-t1,niid-t2,iid-even,iid-uneven missing_scenario_name=mcar,mar-heter,mnar2-heter,mnar2-homog,mar-homog imputer_name=miwae \
fed_strategy_name=local,fedavg,fedavg_ft,fedprox,central round_idx=0,1,4,2,3 eval_params.model=linear,nn log_to_file=False log_level=SUCCESS experiment_name=fed_imp_pc2

# gain
python run_evaluation.py --multirun hydra.launcher.n_jobs=-1 dataset_name=hhip \
data_partition_name=niid-t1,niid-t2,iid-even missing_scenario_name=mcar,mar-heter,mnar2-heter,mnar2-homog,mar-homog imputer_name=gain \
fed_strategy_name=local,fedavg,fedavg_ft,fedprox,central round_idx=0,1,4,2,3 eval_params.model=linear,nn log_to_file=False log_level=SUCCESS experiment_name=fed_imp_pc2
