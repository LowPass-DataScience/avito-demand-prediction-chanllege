import pandas as pd
import numpy as np
import os
import sys
import json
import time
import gc
gc.enable()

dataPath = '../../../data/avito-demand-prediction'

if __name__ == "__main__":
    ss = time.time()
    df_lst = []
    for i in range(15):
        train_active = pd.read_hdf(f'{dataPath}/train_active_h5/train_active_basic_Data_{i}.h5', key='/Raw', mode='r')
        print(f'train_active_{i}.h5 loaded! {time.time()-ss:.2f} s')
        df_lst.append(train_active)
        del train_active
        gc.collect()

    for i in range(13):
        test_active = pd.read_hdf(f'{dataPath}/test_active_h5/test_active_basic_Data_{i}.h5', key='/Raw', mode='r')
        print(f'test_active_{i}.h5 loaded! {time.time()-ss:.2f} s')
        df_lst.append(test_active)
        del test_active
        gc.collect()

    df_active = pd.concat(df_lst, ignore_index=True)
    print(f'Active data combined! {time.time()-ss:.2f} s')

    del df_lst
    gc.collect()

    train_basic = pd.read_hdf(f'{dataPath}/basicData.h5', key='trainRaw', mode='r')
    test_basic = pd.read_hdf(f'{dataPath}/basicData.h5', key='testRaw', mode='r')

    df_all = pd.concat([df_active, train_basic, test_basic], ignore_index=True)
    print(f'Combine all Data! {time.time()-ss:.2f} s')

    df_all.drop_duplicates(['item_id'], inplace=True)
    df_all.reset_index(inplace=True)
    df_all.drop(['index'], axis=1, inplace=True)
    print(f'Dropped duplicates in all data. {time.time()-ss:.2f} s')
    
    del train_basic
    del test_basic
    del df_active
    gc.collect()

    print('Saving data...')
    if 'all_active_data.feather' in os.listdir(f'{dataPath}'):
        os.remove(f'{dataPath}/all_active_data.feather')
        print(f'Remove old all_active_data.feather')

    df_all.to_feather(f'{dataPath}/all_active_data.feather')
    print(f'all_active_data.feather saved! {time.time()-ss:.2f} s')
