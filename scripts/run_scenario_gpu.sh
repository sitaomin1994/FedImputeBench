#!/bin/bash

# Read arguments from the command line
dataset_name=$1
imputer_name=$2
ms_scenario=$3
n_jobs=$4

# Predefined strings for fed_strategy and rounds
fed_strategy='local,fedavg,fedavg_ft,fedprox,central'
rounds='0,1,2,3,4'

if [ "$dataset_name" = "codrna" ]; then
    command="python run_fed_imp_scenario.py --multirun \
        hydra.launcher.n_jobs=$n_jobs \
        dataset_name=$dataset_name \
        imputer=$imputer_name \
        data_partition_name=iid-even,iid-uneven,niid-f1,niid-f2 \
        missing_scenario_name=$ms_scenario \
        fed_strategy=$fed_strategy \
        round_id=$rounds"

elif [ "$dataset_name" = "hhip" ]; then
    command="python run_fed_imp_scenario.py --multirun \
        dataset_name=$dataset_name \
        imputer=$imputer_name \
        data_partition_name=iid-even,iid-uneven,niid-t1,niid-t2 \
        missing_scenario_name=$ms_scenario \
        fed_strategy=$fed_strategy \
        round_id=$rounds"
elif [ "$dataset_name" = "california" ]; then
    command="python run_fed_imp_scenario.py --multirun \
        dataset_name=$dataset_name \
        imputer=$imputer_name \
        data_partition_name=iid-even,iid-uneven,niid-t1,niid-t2 \
        missing_scenario_name=$ms_scenario \
        fed_strategy=$fed_strategy \
        round_id=$rounds"
elif [ "$dataset_name" = "dvisits" ]; then
    command="python run_fed_imp_scenario.py --multirun \
        dataset_name=$dataset_name \
        imputer=$imputer_name \
        data_partition_name=iid-even,iid-uneven,niid-t1,niid-t2,niid-t1,niid-t3 \
        missing_scenario_name=$ms_scenario \
        fed_strategy=$fed_strategy \
        round_id=$rounds"
elif [ "$dataset_name" = "vihecle" ]; then
    command="python run_fed_imp_scenario.py --multirun \
        dataset_name=$dataset_name \
        imputer=$imputer_name \
        data_partition_name=iid-even,iid-uneven,niid-f1,niid-f2 \
        missing_scenario_name=$ms_scenario \
        fed_strategy=$fed_strategy \
        round_id=$rounds"
else
    echo "Error: Unknown dataset name '$dataset_name'"
    exit 1
fi

# Run the command
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
echo $command
eval $command



