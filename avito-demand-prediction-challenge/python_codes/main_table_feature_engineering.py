import pandas as pd
import numpy as np
import os
import gc
from datetime import datetime
gc.enable

dataPath = '../../../data/avito-demand-prediction'

field_lst = ['region', 'category_name', 'parent_category_name']
stat_lst = ['mean', 'std']
def get_cat_dict(df, fields=field_lst, stats=stat_lst):
    tmp_dict = {}
    for f in fields:
        tmpgp = df[[f, 'deal_probability']].groupby(f)
        for s in stats:
            tmp_dict[f+'_'+s] = getattr(tmpgp, s)().to_dict()['deal_probability']
    return tmp_dict

def get_numeric_feature(df, numeric_dict, fields=field_lst, stats=stat_lst):
    for f in fields:
        for s in stats:
            df[f+'_dp_'+s] = df[f].apply(lambda x: numeric_dict[f+'_'+s][x])
    return df

if __name__ == "__main__":
    df_train = pd.read_hdf(f'{dataPath}/basicData.h5', key='trainRaw', mode='r')
    df_test = pd.read_hdf(f'{dataPath}/basicData.h5', key='testRaw', mode='r')
    df_target = pd.read_hdf(f'{dataPath}/basicData.h5', key='trainTarget', mode='r') 
    print(f'basic data loaded!')

    df_train_target = pd.concat([df_train, df_target], axis=1)
    print(f'train_target combined')

    numeric_dict = get_cat_dict(df_train_target)
    df_train_new = get_numeric_feature(df_train, numeric_dict)
    df_test_new = get_numeric_feature(df_test, numeric_dict)
    print(f'New deal_probability features added!')


    df_train_new.drop([c for c in df_train_new.columns if 'dp' not in c and c != 'item_id'], inplace=True, axis=1)
    df_test_new.drop([c for c in df_test_new.columns if 'dp' not in c and c != 'item_id'], inplace=True, axis=1)
    print(f'Dropped columns not needed!')

    del df_train
    del df_test
    del df_target
    del df_train_target
    gc.collect()

    train_all = pd.read_feather(f'{dataPath}/all_data_train.feather')
    test_all = pd.read_feather(f'{dataPath}/all_data_test.feather')
    print(f'Loaded preprocessed data')

    if 'all_data_train_v1.feather' in os.listdir(f'{dataPath}'):
        os.remove(f'{dataPath}/all_data_train_v1.feather')
        print(f'Remove old {dataPath}/all_data_train_v1.feather')
    if 'all_data_test_v1.feather' in os.listdir(f'{dataPath}'):
        os.remove(f'{dataPath}/all_data_test_v1.feather')
        print(f'Remove old {dataPath}/all_data_test_v1.feather')

    train_all_v1 = pd.merge(train_all, df_train_new, how='left', on='item_id')
    test_all_v1 = pd.merge(test_all, df_test_new, how='left', on='item_id')
    print('New features merged!')

    refDate = datetime(2017, 1, 1)
    train_all_v1['activation_weekday'] = train_all_v1.activation_date.apply(lambda x: x.dayofweek)
    test_all_v1['activation_weekday'] = test_all_v1.activation_date.apply(lambda x: x.dayofweek)
    train_all_v1.activation_date = train_all_v1.activation_date.apply(lambda x: (x.to_pydatetime()-refDate).days)
    test_all_v1.activation_date = test_all_v1.activation_date.apply(lambda x: (x.to_pydatetime()-refDate).days)
    print('Datetime columns processed')

    print('Saving data ......')
    train_all_v1.to_feather(f'{dataPath}/all_data_train_v1.feather')
    print(f'{dataPath}/all_data_train_v1.feather SAVED!')
    test_all_v1.to_feather(f'{dataPath}/all_data_test_v1.feather')
    print(f'{dataPath}/all_data_test_v1.feather SAVED!')
