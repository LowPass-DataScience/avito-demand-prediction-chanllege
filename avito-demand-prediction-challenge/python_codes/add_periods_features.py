import pandas as pd
import numpy as np
import os
import sys
import json
import time
import gc
from datetime import datetime, timedelta
gc.enable()

dataPath = '../../../data/avito-demand-prediction'

if __name__ == "__main__":
    ## Load preprocessed combined active data
    ss = time.time()
    active_all = pd.read_feather(f'{dataPath}/all_active_data.feather')
    active_bsc = active_all[['item_id', 'user_id']]
    del active_all
    gc.collect()
    print(f'Active data loaded! {time.time()-ss:.2f} s')

    ## Load raw periods data and merge train/test
    periods_train = pd.read_csv(f'{dataPath}/periods_train.csv.zip', compression='zip', parse_dates=['date_from', 'date_to'])
    periods_test = pd.read_csv(f'{dataPath}/periods_test.csv.zip', compression='zip', parse_dates=['date_from', 'date_to'])
    print(f'Periods data loaded! {time.time()-ss:.2f} s')
    periods_all = pd.concat([periods_train, periods_test], ignore_index=True)
    del periods_train
    del periods_test
    print(f'Periods data merged! {time.time()-ss:.2f} s')

    ## Feature engineering of periods data and merge with active data user_id
    periods_all['days_up'] = (periods_all['date_to'] - periods_all['date_from']) / timedelta(days=1)
    gp = periods_all.groupby('item_id')[['days_up']]
    gp_df = pd.DataFrame()
    gp_df['days_up_sum'] = gp.sum()['days_up']
    gp_df['times_put_up'] = gp.count()['days_up']
    gp_df.reset_index(inplace=True)
    gp_df.rename(index=str, columns={'index': 'item_id'}, inplace=True)
    print(f'Days up from periods collected. {time.time()-ss:.2f} s')

    periods_all.drop_duplicates(['item_id'], inplace=True)
    periods_all = periods_all.merge(gp_df, on='item_id', how='left')
    periods_all = periods_all.merge(active_bsc, on='item_id', how='left')
    del gp
    del gp_df
    gc.collect()
    print(f'Periods data processed! {time.time()-ss:.2f} s')

    ## Group periods data by user_id
    gp_all = periods_all.groupby('user_id')[['days_up_sum', 'times_put_up']] \
                .mean() \
                .reset_index() \
                .rename(index=str, columns={'days_up_sum': 'ave_days_up_user', 'times_put_up': 'avg_times_up_user'})

    n_user_items = active_bsc.groupby('user_id')[['item_id']] \
                .count() \
                .reset_index() \
                .rename(index=str, columns={'item_id': 'n_user_items'})

    gp_all = gp_all.merge(n_user_items, on='user_id', how='outer')
    del active_bsc
    del n_user_items
    gc.collect()
    print(f'Periods data grouped by user_id! {time.time()-ss:.2f} s')

    ## Load main table 
    train_bsc = pd.read_hdf(f'{dataPath}/basicData.h5', key='/trainRaw', mode='r')
    test_bsc = pd.read_hdf(f'{dataPath}/basicData.h5', key='/testRaw', mode='r')
    print(f'Basic data loaded! {time.time()-ss:.2f} s')
    train_ini = train_bsc[['item_id', 'user_id']]
    test_ini = test_bsc[['item_id', 'user_id']]
    del train_bsc
    del test_bsc
    gc.collect()
    print(f'Collected item_id and user_id from main tables. {time.time()-ss:.2f} s')

    ## Merge to main table
    traindf = train_ini.merge(gp_all, how='left', on='user_id')
    testdf = test_ini.merge(gp_all, how='left', on='user_id')

    if f'train_periods_grouped_data.feather' in os.listdir(f'{dataPath}'):
        os.remove(f'{dataPath}/train_periods_grouped_data.feather')
        print(f'Remove old {dataPath}/train_periods_grouped_data.feather')

    if f'test_periods_grouped_data.feather' in os.listdir(f'{dataPath}'):
        os.remove(f'{dataPath}/test_periods_grouped_data.feather')
        print(f'Remove old {dataPath}/test_periods_grouped_data.feather')

    traindf.to_feather(f'{dataPath}/train_periods_grouped_data.feather')
    testdf.to_feather(f'{dataPath}/test_periods_grouped_data.feather')
