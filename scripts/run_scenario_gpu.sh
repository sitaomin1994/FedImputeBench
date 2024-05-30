#!/bin/bash

# Read arguments from the command line
dataset_name=$1
data_partition_name=$2
ms_scenario=$3
imputer_name=$4
fed_strategy=$5
n_jobs=$6

# Predefined strings for fed_strategy and rounds
# fed_strategy='local,fedavg,fedavg_ft,fedprox,central'
rounds='0,1,2,3,4'

command="python run_fed_imp_scenario.py --multirun \
        hydra.launcher.n_jobs=$n_jobs \
        dataset_name=$dataset_name \
        imputer=$imputer_name \
        data_partition_name=$data_partition_name \
        missing_scenario_name=$ms_scenario \
        fed_strategy=$fed_strategy \
        round_id=$rounds \
        experiment.log_to_file=True \
        use_default_hyper_params=True"

# Run the command
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
echo $command
eval $command




