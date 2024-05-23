#!/bin/bash

scenario=$1

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
command="python run_fed_imp_scenario.py --multirun \
hydra.launcher.n_jobs=40 \
dataset_name=hhip \
imputer=em \
data_partition_name=iid-even,iid-uneven,niid-f1,niid-f2,niid-t1,niid-t2 \
missing_scenario_name=$scenario \
fed_strategy=local,central \
round_id=0,1,2,3,4,5,6,7,8,9"

echo $command
eval $command
