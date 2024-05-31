#!/bin/bash
dataset_name=$1
data_partition=$2

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
command="python run_fed_imp_scenario.py --multirun \
hydra.launcher.n_jobs=-1 \
dataset_name=$dataset_name \
imputer=linear_ice \
data_partition_name=$data_partition \
missing_scenario_name=mcar,mar-homog,mar-heter,mnar2-homog,mnar2-heter \
fed_strategy=local,fedavg,central \
round_id=0,1,2,3,4,5,6,7,8,9"

echo $command
eval $command
