#!/bin/bash

# Read arguments from the command line
dataset_name=$1
imputer_name=$2
ms_scenario=$3
fed_strategy=$4
n_jobs=$5

# Predefined strings for fed_strategy and rounds
# fed_strategy='local,fedavg,fedavg_ft,fedprox,central'
rounds='0,1,2,3,4'

if [ "$dataset_name" = "codrna" ]; then

    if [ "$imputer_name" = "gain" ]; then
      hyper_params="imputer.imp_params.imp_model_params.h_dim=8 imputer.imp_params.imp_model_params.loss_alpha=20 \
      imputer.imp_params.imp_model_params.hint_rate=0.5 \
      imputer.model_train_params.local_epoch=5 imputer.model_train_params.global_epoch=300 \
      imputer.model_train_params.weight_decay=0.001"
    elif [ "$imputer_name" = "miwae" ]; then
      hyper_params="imputer.imp_params.imp_model_params.latent_size=8 imputer.imp_params.imp_model_params.n_hidden=32 \
      imputer.model_train_params.local_epoch=10 imputer.model_train_params.global_epoch=500 \
      imputer.model_train_params.weight_decay=0.0001"
    else
      echo "Error: Unknown imputer name '$imputer_name'"
      exit 1
    fi

    data_partition_name="iid-even,iid-uneven,niid-f1,niid-f2"

elif [ "$dataset_name" = "hhip" ]; then

    if [ "$imputer_name" = "gain" ]; then
      hyper_params="imputer.imp_params.imp_model_params.h_dim=32 imputer.imp_params.imp_model_params.loss_alpha=100 \
      imputer.imp_params.imp_model_params.hint_rate=0.5 \
      imputer.model_train_params.local_epoch=5 imputer.model_train_params.global_epoch=600 \
      imputer.model_train_params.weight_decay=0.01"
    elif [ "$imputer_name" = "miwae" ]; then
      hyper_params="imputer.imp_params.imp_model_params.latent_size=32 imputer.imp_params.imp_model_params.n_hidden=32 \
      imputer.model_train_params.local_epoch=10 imputer.model_train_params.global_epoch=500 \
      imputer.model_train_params.weight_decay=0.0001"
    else
      echo "Error: Unknown imputer name '$imputer_name'"
      exit 1
    fi
    data_partition_name="iid-even,iid-uneven,niid-t1,niid-t2"

elif [ "$dataset_name" = "california" ]; then

    if [ "$imputer_name" = "gain" ]; then
      hyper_params="imputer.imp_params.imp_model_params.h_dim=8 imputer.imp_params.imp_model_params.loss_alpha=20 \
      imputer.imp_params.imp_model_params.hint_rate=0.5 \
      imputer.model_train_params.local_epoch=5 imputer.model_train_params.global_epoch=300 \
      imputer.model_train_params.weight_decay=0.001"
    elif [ "$imputer_name" = "miwae" ]; then
      hyper_params="imputer.imp_params.imp_model_params.latent_size=8 imputer.imp_params.imp_model_params.n_hidden=32 \
      imputer.model_train_params.local_epoch=10 imputer.model_train_params.global_epoch=500 \
      imputer.model_train_params.weight_decay=0.0001"
    else
      echo "Error: Unknown imputer name '$imputer_name'"
      exit 1
    fi
    data_partition_name="iid-even,iid-uneven,niid-t1,niid-t2"

elif [ "$dataset_name" = "dvisits" ]; then

    if [ "$imputer_name" = "gain" ]; then
      hyper_params="imputer.imp_params.imp_model_params.h_dim=8 imputer.imp_params.imp_model_params.loss_alpha=20 \
      imputer.imp_params.imp_model_params.hint_rate=0.5 \
      imputer.model_train_params.local_epoch=5 imputer.model_train_params.global_epoch=300 \
      imputer.model_train_params.weight_decay=0.001"
    elif [ "$imputer_name" = "miwae" ]; then
      hyper_params="imputer.imp_params.imp_model_params.latent_size=8 imputer.imp_params.imp_model_params.n_hidden=128 \
      imputer.model_train_params.local_epoch=10 imputer.model_train_params.global_epoch=300 \
      imputer.model_train_params.weight_decay=0.001"
    else
      echo "Error: Unknown imputer name '$imputer_name'"
      exit 1
    fi
    data_partition_name="iid-even,iid-uneven,niid-t1,niid-t2"

elif [ "$dataset_name" = "vehicle" ]; then

    if [ "$imputer_name" = "gain" ]; then
      hyper_params="imputer.imp_params.imp_model_params.h_dim=32 imputer.imp_params.imp_model_params.loss_alpha=20 \
      imputer.imp_params.imp_model_params.hint_rate=0.5 \
      imputer.model_train_params.local_epoch=5 imputer.model_train_params.global_epoch=300 \
      imputer.model_train_params.weight_decay=0.001"
    elif [ "$imputer_name" = "miwae" ]; then
      hyper_params="imputer.imp_params.imp_model_params.latent_size=16 imputer.imp_params.imp_model_params.n_hidden=64 \
      imputer.model_train_params.local_epoch=5 imputer.model_train_params.global_epoch=300 \
      imputer.model_train_params.weight_decay=0.001"
    else
      echo "Error: Unknown imputer name '$imputer_name'"
      exit 1
    fi
    data_partition_name="iid-even,iid-uneven,niid-f1,niid-f2"

elif [ "$dataset_name" = "school_pca" ]; then

    hyper_params=""
    data_partition_name="iid-even,iid-uneven,niid-f1,niid-f2"

else
    echo "Error: Unknown dataset name '$dataset_name'"
    exit 1
fi

command="python run_fed_imp_scenario.py --multirun \
        hydra.launcher.n_jobs=$n_jobs \
        dataset_name=$dataset_name \
        imputer=$imputer_name \
        data_partition_name=$data_partition_name \
        missing_scenario_name=$ms_scenario \
        fed_strategy=$fed_strategy \
        round_id=$rounds \
        experiment.log_to_file=True \
        $hyper_params"

# Run the command
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
echo $command
eval $command



