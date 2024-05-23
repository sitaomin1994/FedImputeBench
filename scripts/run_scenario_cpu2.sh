#!/bin/bash

scenario=$1

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
command="python run_fed_imp_scenario.py --multirun \
hydra.launcher.n_jobs=10 \
dataset_name=hhip \
imputer=missforest \
data_partition_name=niid-f1 \
missing_scenario_name=$scenario \
fed_strategy=local,central \
round_id=0,1,2,3,4,5,6,7,8,9"

echo $command
eval $command
