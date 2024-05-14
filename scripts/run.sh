export CUBLAS_WORKSPACE_CONFIG=":4096:8"
python main.py --multirun \
data_partition=iid,iid2,niid1,niid2,niidf1,niidf2 \
missing_scenario=mcar,mar2_homo,mar2_homo_g,mar2_heter \
imputer=simple,linear_ice \
fed_strategy=local,fedavg,central

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
python main.py --multirun \
data_partition=iid,iid2,niid1,niid2,niidf1,niidf2 \
missing_scenario=mar_homo_g,mar_homo,mar_heter,mnar_homo,mnar_homo_g,mnar_heter \
imputer=simple,linear_ice \
fed_strategy=local,fedavg,central

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
python main.py --multirun \
data_partition=iid,iid2,niid1,niid2,niidf1,niidf2 \
missing_scenario=mcar,mar2_homo,mar2_homo_g,mar2_heter \
imputer=miwae \
fed_strategy=local,fedavg,fedavg_ft

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
python main.py --multirun \
data_partition=iid,iid2,niid1,niid2,niidf1,niidf2 \
missing_scenario=mar_homo_g,mar_homo,mar_heter,mnar_homo,mnar_homo_g,mnar_heter \
imputer=miwae \
fed_strategy=local,fedavg,fedavg_ft

#export CUBLAS_WORKSPACE_CONFIG=":4096:8"
#python main.py --multirun \
#data_partition=iid,niidf2 \
#missing_scenario=mcar,mar2_homo,mar2_homo_g,mar_homo,mar_homo_g,mnar_homo,mar_heter,mnar_heter \
#imputer=miwae \
#fed_strategy=local,fedavg,fedavg_ft

#export CUBLAS_WORKSPACE_CONFIG=":4096:8"
#python main.py --multirun \
#data_partition=iid,iid2,niid1,niid2 \
#missing_scenario=mar2_homo_g \
#imputer=simple,linear_ice \
#fed_strategy=local,fedavg,central


