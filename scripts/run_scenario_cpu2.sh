#!/bin/bash

scenario=$1

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
command="python run_fed_imp_scenario.py --multirun \
hydra.launcher.n_jobs=24 \
dataset_name=vehicle \
imputer=missforest \
data_partition_name=iid-even,iid-uneven,niid-f1,niid-f2 \
missing_scenario_name=mnar2-homog \
fed_strategy=local,fedtree,central \
round_id=0,1,2,3,4,5,6,7,8,9"

echo $command
eval $command

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
command="python run_fed_imp_scenario.py --multirun \
hydra.launcher.n_jobs=24 \
dataset_name=hhip \
imputer=missforest \
data_partition_name=iid-uneven \
missing_scenario_name=mnar1-homog \
fed_strategy=local,fedtree,central \
round_id=0,1,2,3,4,5,6,7,8,9"

echo $command
eval $command
