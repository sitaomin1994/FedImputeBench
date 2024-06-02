#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

python run_evaluation.py --multirun hydra.launcher.n_jobs=-1 dataset_name=hhip \
data_partition_name=niid-t2,niid-t1 missing_scenario_name=mcar,mar-heter,mar-homog,mnar2-heter,mnar2-homog imputer_name=missforest fed_strategy_name=fedtree,central,local round_idx=7,0,5,2,4,9,8,6,3,1 eval_params.model=linear,nn log_to_file=False log_level=SUCCESS experiment_name=fed_imp_pc2