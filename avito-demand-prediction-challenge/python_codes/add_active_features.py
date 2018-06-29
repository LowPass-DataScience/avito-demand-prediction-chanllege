import pandas as pd
import numpy as np
import os
import sys
import json
import time
import gc
gc.enable()

dataPath = '../../../data/avito-demand-prediction'

def add_features_from_active(df_train, df_test, df_act, field, gpname, newname, navl, stat):
    """
    df_train, df_test: main tables
    df_act: active table
    field: features to engineer
    gpname: feature to group
    newname: new name for the feature after engineering
    navl: values to replace all null 
    stat: built-in stats operation, eg: mean, std, nunique, etc
    """
    tmpgp = df_act[[gpname, field]].groupby(gpname)
    tmpdf = getattr(tmpgp[field], stat)().fillna(navl)
    tmpdf = tmpdf.reset_index(gpname)
    print(f'{field} grouped by {gpname} has completed. {time.time()-ss:.2f} s')

    tmpdf.rename(index=str, columns={field: newname}, inplace=True)
    print(f'Rename {field} to {newname}. {time.time()-ss:.2f} s')

    df_train_out = pd.merge(df_train, tmpdf, how='left', on=gpname)
    df_test_out = pd.merge(df_test, tmpdf, how='left', on=gpname)
    print(f'Merged features to main table. {time.time()-ss:.2f} s')

    del tmpgp
    del tmpdf
    del df_train
    del df_test
    del df_act
    gc.collect()

    return df_train_out, df_test_out
    
if __name__ == "__main__":
    ss = time.time()
    active_all = pd.read_feather(f'{dataPath}/all_active_data.feather')
    print(f'Active data loaded! {time.time()-ss:.2f} s')

    train_bsc = pd.read_hdf(f'{dataPath}/basicData.h5', key='/trainRaw', mode='r')
    test_bsc = pd.read_hdf(f'{dataPath}/basicData.h5', key='/testRaw', mode='r')
    print(f'Basic data loaded! {time.time()-ss:.2f} s')

    train_ini = train_bsc[['item_id', 'user_id', 'category_name', 'parent_category_name', 'price']]
    test_ini = test_bsc[['item_id', 'user_id', 'category_name', 'parent_category_name', 'price']]

    del train_bsc
    del test_bsc
    gc.collect()
    print(f'Collected item_id and user_id from main tables. {time.time()-ss:.2f} s')

    traindf, testdf = add_features_from_active(df_train = train_ini, 
                                               df_test = test_ini,
                                               df_act = active_all, 
                                               field = 'region', 
                                               gpname = 'user_id',
                                               newname = 'region_user_nunique',
                                               navl = 0, 
                                               stat = 'nunique')

    traindf, testdf = add_features_from_active(df_train = traindf, 
                                               df_test = testdf,
                                               df_act = active_all, 
                                               field = 'city', 
                                               gpname = 'user_id',
                                               newname = 'city_user_nunique',
                                               navl = 0, 
                                               stat = 'nunique')

    traindf, testdf = add_features_from_active(df_train = traindf, 
                                               df_test = testdf,
                                               df_act = active_all, 
                                               field = 'category_name', 
                                               gpname = 'user_id',
                                               newname = 'ctg_name_user_nunique',
                                               navl = 0, 
                                               stat = 'nunique')

    traindf, testdf = add_features_from_active(df_train = traindf, 
                                               df_test = testdf,
                                               df_act = active_all, 
                                               field = 'parent_category_name', 
                                               gpname = 'user_id',
                                               newname = 'prtctg_nm_user_nunique',
                                               navl = -1, 
                                               stat = 'nunique')

    traindf, testdf = add_features_from_active(df_train = traindf, 
                                               df_test = testdf,
                                               df_act = active_all, 
                                               field = 'price', 
                                               gpname = 'user_id',
                                               newname = 'price_user_mean',
                                               navl = -1, 
                                               stat = 'mean')

    traindf, testdf = add_features_from_active(df_train = traindf, 
                                               df_test = testdf,
                                               df_act = active_all, 
                                               field = 'price', 
                                               gpname = 'user_id',
                                               newname = 'price_user_std',
                                               navl = 0, 
                                               stat = 'std')
    
    traindf, testdf = add_features_from_active(df_train = traindf, 
                                               df_test = testdf,
                                               df_act = active_all, 
                                               field = 'price', 
                                               gpname = 'category_name',
                                               newname = 'price_ctg_mean',
                                               navl = -1, 
                                               stat = 'mean')

    traindf, testdf = add_features_from_active(df_train = traindf, 
                                               df_test = testdf,
                                               df_act = active_all, 
                                               field = 'price', 
                                               gpname = 'category_name',
                                               newname = 'price_ctg_std',
                                               navl = 0, 
                                               stat = 'std')

    traindf, testdf = add_features_from_active(df_train = traindf, 
                                               df_test = testdf,
                                               df_act = active_all, 
                                               field = 'price', 
                                               gpname = 'parent_category_name',
                                               newname = 'price_prtctg_mean',
                                               navl = -1, 
                                               stat = 'mean')

    traindf, testdf = add_features_from_active(df_train = traindf, 
                                               df_test = testdf,
                                               df_act = active_all, 
                                               field = 'price', 
                                               gpname = 'parent_category_name',
                                               newname = 'price_prtctg_std',
                                               navl = 0, 
                                               stat = 'std')

    traindf['price_diff_ctg_ratio'] = (traindf.price - traindf.price_ctg_mean) / traindf.price_ctg_std
    traindf['price_diff_prtctg_ratio'] = (traindf.price - traindf.price_prtctg_mean) / traindf.price_prtctg_std
    testdf['price_diff_ctg_ratio'] = (testdf.price - testdf.price_ctg_mean) / testdf.price_ctg_std
    testdf['price_diff_prtctg_ratio'] = (testdf.price - testdf.price_prtctg_mean) / testdf.price_prtctg_std

    if f'train_active_grouped_data.feather' in os.listdir(f'{dataPath}'):
        os.remove(f'{dataPath}/train_active_grouped_data.feather')
        print(f'Remove old {dataPath}/train_active_grouped_data.feather')

    if f'test_active_grouped_data.feather' in os.listdir(f'{dataPath}'):
        os.remove(f'{dataPath}/test_active_grouped_data.feather')
        print(f'Remove old {dataPath}/test_active_grouped_data.feather')

    traindf.to_feather(f'{dataPath}/train_active_grouped_data.feather')
    testdf.to_feather(f'{dataPath}/test_active_grouped_data.feather')
