#!/bin/bash

# Read arguments from the command line
dataset_name=$1
dp=$2
imputer=$3
fed_strategy=$4
eval_model=$5
n_jobs=$6
experiment_name=$7

if [ $imputer != "miwae" ] && [ $imputer != "gain" ]; then
    rounds="0,1,2,3,4,5,6,7,8,9"
else
    rounds="0,1,2,3,4"
fi

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
command="python run_evaluation.py --multirun \
hydra.launcher.n_jobs=$n_jobs \
dataset_name=$dataset_name \
imputer_name=$imputer \
data_partition_name=$dp \
missing_scenario_name=mcar,mar-homog,mar-heter,mnar1-homog,mnar1-heter,mnar2-homog,mnar2-heter \
fed_strategy_name=$fed_strategy \
round_idx=$rounds \
eval_params.model=$eval_model \
log_to_file=False experiment_name=$experiment_name"

echo $command
eval $command

