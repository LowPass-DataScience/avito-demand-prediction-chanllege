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
    ## Load raw data
    traindf = pd.read_csv(f'{dataPath}/train.csv.zip', compression='zip', usecols=['item_id', 'param_1', 'param_2', 'param_3']) 
    testdf = pd.read_csv(f'{dataPath}/test.csv.zip', compression='zip', usecols=['item_id', 'param_1', 'param_2', 'param_3']) 
    ff = time.time()
    print(f'Train and Test raw data loaded! {ff-ss:.2f} s')

    traindf['param_all'] = traindf.param_1.fillna('') + ' ' + traindf.param_2.fillna('') + ' ' +traindf.param_3.fillna('')
    testdf['param_all'] = testdf.param_1.fillna('') + ' ' + testdf.param_2.fillna('') + ' ' +testdf.param_3.fillna('')

    traindf.drop(['param_1', 'param_2', 'param_3'], inplace=True, axis=1)
    testdf.drop(['param_1', 'param_2', 'param_3'], inplace=True, axis=1)

    ## Apply LSA for title info
    param_lsa = lf.LSA_app(conf='param_all')
    param_lsa.get_tfidf(traindf, testdf)
    ff = time.time()
    print(f'finshed tfidf setup for param_all: {ff-ss:.2f} s')

    param_lsa.apply_svd()
    ff = time.time()
    print(f'finshed svd setup for param_all: {ff-ss:.2f} s')

    train_tfidf = param_lsa.get_df_tfidf(traindf)
    test_tfidf = param_lsa.get_df_tfidf(testdf)
    train_lsa_df = param_lsa.get_svd(traindf, train_tfidf)
    test_lsa_df = param_lsa.get_svd(testdf, test_tfidf)
    
    del train_tfidf
    del test_tfidf
    del param_lsa
    gc.collect()
    
    ff = time.time()
    print(f'finshed svd features for param_all: {ff-ss:.2f} seconds')
    
    ## Merge to basic dataframes
    train_lsa_df.drop(['param_all'], axis=1, inplace=True)
    test_lsa_df.drop(['param_all'], axis=1, inplace=True)


    ## Load preprocessed data
    train_v3 = pd.read_feather(f'{dataPath}/features/all_data_train_v3.feather')
    test_v3 = pd.read_feather(f'{dataPath}/features/all_data_test_v3.feather')
    ff = time.time()
    print(f'Loaded train_v3 and test_v3. {ff-ss:.2f} s')

    if 'all_data_train_v4.feather' in os.listdir(f'{dataPath}/features'):
        os.remove(f'{dataPath}/features/all_data_train_v4.feather')
        print(f'Remove old all_data_train_v4.feather')

    if 'all_data_test_v4.feather' in os.listdir(f'{dataPath}/features'):
        os.remove(f'{dataPath}/features/all_data_test_v4.feather')
        print(f'Remove old all_data_test_v4.feather')

    train_v4 = train_v3.merge(train_lsa_df, how='left', on='item_id')
    test_v4 = test_v3.merge(test_lsa_df, how='left', on='item_id')
    ff = time.time()
    print(f'Data_v4 merged, {ff-ss:.2f} s')

    train_v4.to_feather(f'{dataPath}/features/all_data_train_v4.feather')
    test_v4.to_feather(f'{dataPath}/features/all_data_test_v4.feather')
    ff = time.time()
    print(f'Data_v4 saved, {ff-ss:.2f} s')
