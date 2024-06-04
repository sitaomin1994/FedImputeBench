#!/bin/bash

#./scripts/run_eval_fed.sh hhip iid-even,iid-uneven,niid-t1,niid-t2 mcar,mar-homog,mar-heter,mnar2-homog,mnar2-heter \
#miwae local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
#./scripts/run_eval_fed.sh hhip iid-even,iid-uneven,niid-t1,niid-t2 mcar,mar-homog,mar-heter,mnar2-homog,mnar2-heter \
#gain local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
#./scripts/run_eval_fed.sh hhip iid-even,iid-uneven,niid-t1,niid-t2 mcar,mar-homog,mar-heter,mnar2-homog,mnar2-heter \
#missforest local,fedtree nn -1 fed_imp_pc2
#./scripts/run_eval_fed.sh hhip iid-even,iid-uneven,niid-t1,niid-t2 mcar,mar-homog,mar-heter,mnar2-homog,mnar2-heter \
#simple,em,linear_ice local,fedavg nn -1 fed_imp_pc2

export CUBLAS_WORKSPACE_CONFIG=":4096:8"

python run_evaluation.py -m dataset_name=hhip hydra.launcher.n_jobs=-1 \
data_partition_name=iid-even missing_scenario_name=mcar imputer_name=miwae fed_strategy_name=local,fedavg_ft round_idx=0,1,2,3,4 eval_params.model=nn experiment_name=fed_imp_pc2
python run_evaluation.py -m dataset_name=hhip hydra.launcher.n_jobs=-1 \
data_partition_name=iid-even missing_scenario_name=mar-homog imputer_name=miwae fed_strategy_name=local,fedavg,fedavg_ft round_idx=0,1,2,3,4 eval_params.model=nn experiment_name=fed_imp_pc2
python run_evaluation.py -m dataset_name=hhip hydra.launcher.n_jobs=-1 \
data_partition_name=iid-even missing_scenario_name=mnar2-homog imputer_name=miwae fed_strategy_name=fedavg_ft,fedprox round_idx=0,1,2,3,4 eval_params.model=nn experiment_name=fed_imp_pc2
python run_evaluation.py -m dataset_name=hhip hydra.launcher.n_jobs=-1 \
data_partition_name=iid-uneven missing_scenario_name=mnar2-homog imputer_name=miwae fed_strategy_name=fedavg,fedprox,fedavg_ft round_idx=0,1,2,3,4 eval_params.model=nn experiment_name=fed_imp_pc2
python run_evaluation.py -m dataset_name=hhip hydra.launcher.n_jobs=-1 \
data_partition_name=iid-uneven missing_scenario_name=mnar2-heter imputer_name=miwae fed_strategy_name=local,fedavg,fedavg_ft round_idx=0,1,2,3,4 eval_params.model=nn experiment_name=fed_imp_pc2
python run_evaluation.py -m dataset_name=hhip hydra.launcher.n_jobs=-1 \
data_partition_name=niid-t1 missing_scenario_name=mar-homog imputer_name=miwae fed_strategy_name=fedavg_ft,fedprox round_idx=0,1,2,3,4 eval_params.model=nn experiment_name=fed_imp_pc2
python run_evaluation.py -m dataset_name=hhip hydra.launcher.n_jobs=-1 \
data_partition_name=niid-t1 missing_scenario_name=mar-heter imputer_name=miwae fed_strategy_name=fedavg_ft,fedprox round_idx=0,1,2,3,4 eval_params.model=nn experiment_name=fed_imp_pc2
python run_evaluation.py -m dataset_name=hhip hydra.launcher.n_jobs=-1 \
data_partition_name=niid-t1 missing_scenario_name=mnar2-heter imputer_name=miwae fed_strategy_name=local,fedavg_ft round_idx=0,1,2,3,4 eval_params.model=nn experiment_name=fed_imp_pc2
python run_evaluation.py -m dataset_name=hhip hydra.launcher.n_jobs=-1 \
data_partition_name=niid-t2 missing_scenario_name=mnar2-homog imputer_name=miwae fed_strategy_name=local,fedavg,fedavg_ft,fedprox round_idx=0,1,2,3,4 eval_params.model=nn experiment_name=fed_imp_pc2
python run_evaluation.py -m dataset_name=hhip hydra.launcher.n_jobs=-1 \
data_partition_name=niid-t2 missing_scenario_name=mar-heter imputer_name=miwae fed_strategy_name=local,fedavg round_idx=0,1,2,3,4 eval_params.model=nn experiment_name=fed_imp_pc2
