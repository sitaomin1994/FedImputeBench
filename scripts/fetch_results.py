import json
import itertools
import os

import numpy as np
import pandas as pd

data_partition_strategies = [
    'iid-even', 'iid-uneven10range', 'iid-uneven10hs', 'iid-uneven10hsl'
]

missing_simulate = [
    'fixed2@mr=0.5-mm=mcar',
    'fixed@mr=0.5-mm=mnar_quantile-mfunc=lr-k=0.1',
    'fixed@mr=0.5-mm=mnar_quantile-mfunc=lr-k=0.3',
    'fixed@mr=0.5-mm=mnar_quantile-mfunc=lr-k=0.5',
    'fixed@mr=0.5-mm=mnar_quantile-mfunc=lr-k=0.7',
    'fixed@mr=0.5-mm=mnar_quantile-mfunc=lr-k=0.9',
    'fixed2@mr=0.5-mm=mnar_quantile-mfunc=lr-k=0.1',
    'fixed2@mr=0.5-mm=mnar_quantile-mfunc=lr-k=0.3',
    'fixed2@mr=0.5-mm=mnar_quantile-mfunc=lr-k=0.5',
    'fixed2@mr=0.5-mm=mnar_quantile-mfunc=lr-k=0.7',
    'fixed2@mr=0.5-mm=mnar_quantile-mfunc=lr-k=0.9',
]

dataset = ['codon']


dir = './results/raw_results/'

combination = list(itertools.product(dataset, data_partition_strategies, missing_simulate))

df_records = []

for setting in combination:
    dataset, data_partition, missing = setting
    file_name = f"{dataset}/{data_partition}/{missing}/miwae_fedavg.json"
    with open(dir + file_name, 'r') as f:
        data = json.load(f)

    ret = data['rets']
    ret_rmse = []
    ret_ws = []

    for client_id in ret.keys():
        ret_rmse.append(ret[client_id]['imp_rmse'])
        ret_ws.append(ret[client_id]['imp_sliced-ws'])

    df_records.append({
        'dataset': dataset,
        'data_partition': data_partition,
        'missing': missing,
        'rmse': np.array(ret_rmse).mean(),
        'ws': np.array(ret_ws).mean(),
        'rmse_std': np.array(ret_rmse).mean(),
        'ws_std': np.array(ret_ws).mean(),
        'rmse_ind': ret_rmse,
        'ws_ind': ret_ws
    })



df = pd.DataFrame(df_records)
df.columns = ['dataset', 'data_partition', 'missing', 'rmse', 'ws', 'rmse_std', 'ws_std', 'rmse_ind', 'ws_ind']
save_dir = "./results/processed_results/"

with pd.ExcelWriter(f'{save_dir}/results_miwae_favg.xlsx') as writer:
    df.to_excel(writer, sheet_name='avg', index=False)
