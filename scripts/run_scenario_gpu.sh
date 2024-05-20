#!/bin/bash

# Read arguments from the command line
dataset_name=$1
imputer_name=$2
mech_type=$3
n_jobs=$4

# Predefined strings for fed_strategy and rounds
fed_strategy='local,fedavg,fedavg_ft,fedprox'
rounds='0,1,2,3,4'

# Determine the missing scenario based on the mechanical type
if [ "$mech_type" = "heter" ]; then
    ms_scenario='mar-heter,mnar1-heter,mnar2-heter'
elif [ "$mech_type" = "homo" ]; then
    ms_scenario='mcar,mar-homog,mnar1-homog,mnar2-homog'
else
    echo "Error: unknown mech type '$mech_type'"
    exit 1
fi

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
else
    echo "Error: Unknown dataset name '$dataset_name'"
    exit 1
fi

# Run the command
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
echo $command
eval $command



