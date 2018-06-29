import pandas as pd
import numpy as np
import os
import LSA_features as lf
import gc
from multiprocessing import Pool, cpu_count, Array
import time
gc.enable()

# Disable some warnings
import warnings
warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)

dataPath = '../../../data/avito-demand-prediction'

if __name__ == "__main__":
    ss = time.time()
    ## Load preprocessed data
    train_v4 = pd.read_feather(f'{dataPath}/features/all_data_train_v4.feather')
    test_v4 = pd.read_feather(f'{dataPath}/features/all_data_test_v4.feather')
    print(f'Loaded train_v4 and test_v4. {time.time()-ss:.2f} s')

    train_active = pd.read_feather(f'{dataPath}/train_active_grouped_data.feather')
    test_active = pd.read_feather(f'{dataPath}/test_active_grouped_data.feather')
    print(f'Loaded train_active and test_active. {time.time()-ss:.2f} s')

    train_periods = pd.read_feather(f'{dataPath}/train_periods_grouped_data.feather')
    test_periods = pd.read_feather(f'{dataPath}/test_periods_grouped_data.feather')
    print(f'Loaded train_periods and test_periods. {time.time()-ss:.2f} s')

    ## Merge periods data
    train_periods.drop(['user_id'], axis=1, inplace=True)
    test_periods.drop(['user_id'], axis=1, inplace=True)
    train_periods.ave_days_up_user.fillna(0, inplace=True)
    train_periods.avg_times_up_user.fillna(0, inplace=True)
    test_periods.ave_days_up_user.fillna(0, inplace=True)
    test_periods.avg_times_up_user.fillna(0, inplace=True)

    train_v5 = train_v4.merge(train_periods, how='left', on='item_id')
    test_v5 = test_v4.merge(test_periods, how='left', on='item_id')
    print(f'Merged train_periods and test_periods. {time.time()-ss:.2f} s')

    ## Merge active data
    train_active.drop(['user_id', 'category_name', 'parent_category_name', 'price'], axis=1, inplace=True)
    test_active.drop(['user_id', 'category_name', 'parent_category_name', 'price'], axis=1, inplace=True)

    train_v5 = train_v5.merge(train_active, how='left', on='item_id')
    test_v5 = test_v5.merge(test_active, how='left', on='item_id')
    print(f'Merged train_active and test_active. {time.time()-ss:.2f} s')

    if 'all_data_train_v5.feather' in os.listdir(f'{dataPath}/features'):
        os.remove(f'{dataPath}/features/all_data_train_v5.feather')
        print(f'Remove old all_data_train_v5.feather')

    if 'all_data_test_v5.feather' in os.listdir(f'{dataPath}/features'):
        os.remove(f'{dataPath}/features/all_data_test_v5.feather')
        print(f'Remove old all_data_test_v5.feather')

    train_v5.to_feather(f'{dataPath}/features/all_data_train_v5.feather')
    test_v5.to_feather(f'{dataPath}/features/all_data_test_v5.feather')
    print(f'Data_v5 saved, {time.time()-ss:.2f} s')
