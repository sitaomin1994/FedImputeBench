export CUBLAS_WORKSPACE_CONFIG=":4096:8"
python run_fed_imp_scenario.py --multirun \
dataset_name=codrna \
data_partition=iid-even,iid-uneven,niid-t1,niid-t2,niid-f1,niid-f2 \
missing_scenario=mcar,mar-homog,mar-heter,mnar1-homog,mnar1-heter,mnar2-homog,mnar2-heter \
imputer=linear_ice \
fed_strategy=local,fedavg,central \
round_id=0,1,2,3,4,5,6,7,8,9

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
python run_fed_imp_scenario.py --multirun \
data_partition=iid-even,iid-uneven,niid-t1,niid-t2,niid-f1,niid-f2 \
missing_scenario=mcar,mar-homog,mar-heter,mnar1-homog,mnar1-heter,mnar2-homog,mnar2-heter \
imputer=missforest \
fed_strategy=local,fedtree,central


