#export CUBLAS_WORKSPACE_CONFIG=":4096:8"
#python run_fed_imp_scenario.py --multirun \
#dataset_name=codrna \
#imputer=simple \
#data_partition_name=iid-even,iid-uneven,niid-t1,niid-t2,niid-f1,niid-f2 \
#missing_scenario_name=mcar,mar-homog,mar-heter,mnar1-homog,mnar1-heter,mnar2-homog,mnar2-heter \
#fed_strategy=local,fedavg,central \
#round_id=0,1,2,3,4,5,6,7,8,9

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
python run_fed_imp_scenario.py --multirun \
dataset_name=codrna \
imputer=gain \
data_partition_name=iid-even \
missing_scenario_name=mcar,mar-homog \
fed_strategy=fedavg \
round_id=0,1,2,3,4,5,6,7,8,9


