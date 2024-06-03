#!/bin/bash

# Read arguments from the command line
dataset_name=$1
dp=$2
missing=$3
imputer=$4
fed_strategy=$5
eval_model=$6
n_jobs=$7
experiment_name=$8

if [ $imputer != "miwae" ] && [ $imputer != "gain" ]; then
    rounds="0,1,2,3,4,5,6,7,8,9"
else
    rounds="0,1,2,3,4"
fi

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
command="python run_evaluation_fed.py --multirun \
hydra.launcher.n_jobs=$n_jobs \
dataset_name=$dataset_name \
imputer_name=$imputer \
data_partition_name=$dp \
missing_scenario_name=$missing \
fed_strategy_name=$fed_strategy \
round_idx=$rounds \
eval_params.model=$eval_model \
log_to_file=False log_level=SUCCESS experiment_name=$experiment_name"

echo $command
eval $command

