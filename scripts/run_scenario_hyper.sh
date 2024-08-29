#!/bin/bash

# Read arguments from the command line
dataset_name=$1
dp=$2
scenario=$3
imputer=$4
fed_strategy=$5
n_jobs=$6

#export CUBLAS_WORKSPACE_CONFIG=":4096:8"
#command1="python run_fed_imp_scenario.py --multirun \
#hydra.launcher.n_jobs=$n_jobs \
#dataset_name=$dataset_name \
#imputer=simple,em,linear_ice \
#data_partition_name=iid-even,iid-uneven,niid-f1,niid-f2 \
#missing_scenario_name=$scenario \
#fed_strategy=local,fedavg,central \
#round_id=0,1,2,3,4,5,6,7,8,9"
if [ $imputer == "miwae" ]; then
    epoch=10
elif [ $imputer == "gain" ]; then
    epoch=5
else
    epoch=5
fi

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
command2="python run_fed_imp_scenario.py --multirun \
hydra.launcher.n_jobs=$n_jobs \
dataset_name=$dataset_name \
imputer=$imputer \
data_partition_name=$dp \
missing_scenario_name=$scenario \
fed_strategy=$fed_strategy \
round_id=0,1,2,3,4 \
imputer.model_train_params.learning_rate=0.001,0.01,0.1,1,10 \
imputer.model_train_params.local_epoch=$epoch \
imputer.model_train_params.optimizer=adamw \
experiment.log_to_file=True"

#echo $command1
#eval $command1

echo $command2
eval $command2
