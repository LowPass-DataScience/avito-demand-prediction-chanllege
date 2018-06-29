import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import time
import gc
import os
import sys
import json
from multiprocessing import Pool, cpu_count
gc.enable()

randSeed = 1
np.random.seed(randSeed)

## Data path
dataPath = '../../../data/avito-demand-prediction'
resultPath = f'{dataPath}/features'
    
## Get new Version 
def getNewVersion():
    ld = len([f for f in os.listdir('final_results') if 'lowpass' in f])
    maxVer = 0
    for f in ld:
        ver = int(''.join([c for c in f if c.isnumeric()]))
        maxVer = max(maxVer, ver)
    maxVer = str(maxVer+1)
    major = maxVer[0]
    return f'v{major}'

## GBDT parameters
params = {
    'task': 'train',
    'device' : 'cpu',
    'nthread': 64,    # [CPU] number of OpenMP threads
    'tree_learner' : 'data',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'num_leaves': 300,
    'metric': 'rmse',
#    'reg_alpha': 0.1,
#    'reg_lambda': 1.5,
#    'max_depth': 9,
    'learning_rate': 0.020,
#    'max_bin': 255,
#    'min_split_gain': .01,
#    'min_child_samples': 16,
    'subsample': 0.75,
    'subsample_freq': 3,
    'colsample_bytree': 0.9,
    'verbose': 0
}

catFeatures = ['region', 'city', 'parent_category_name', 'param_1', 'param_2', 'param_3', 
               'user_type', 'activation_weekday', 'region_city']

if __name__ == "__main__":
    try:
        rs_idx = int(sys.argv[1])
    except:
        print('Please enter a random seed index')
        exit()
        
    vers = 'v5'
    ss = time.time()
    data_all = pd.read_feather(f'{resultPath}/all_data_train_{vers}.feather')
    target_all = pd.read_hdf(f'{dataPath}/basicData.h5', key='trainTarget', mode='r')
    print('Dataset loaded')
    ff = time.time()
    print(f'Time elapsed! {ff-ss:.2f} seconds')

    zero_idx = np.array(target_all[target_all == 0].index)
    nonzero_idx = np.array(target_all[target_all != 0].index)
    np.random.shuffle(zero_idx)
    np.random.shuffle(nonzero_idx)
    zeroCount = len(zero_idx)
    nonzeroCount = len(nonzero_idx)
    zerosplit = round(zeroCount * 0.2)
    nonzerosplit = round(nonzeroCount * 0.2)

    train_idx_lst = np.concatenate([zero_idx[zerosplit:], nonzero_idx[nonzerosplit:]])
    test_idx_lst = np.concatenate([zero_idx[:zerosplit], nonzero_idx[:nonzerosplit]])

    dataTrain = data_all.loc[train_idx_lst]
    dataTest = data_all.loc[test_idx_lst]
    target = target_all.loc[train_idx_lst]
    target_val = target_all.loc[test_idx_lst]

    ## Final processing
    # Remove invalid index columns
    dataTrain.drop('item_id', axis=1, inplace=True)
    dataTest.drop('item_id', axis=1, inplace=True)
    imageFeat = [col for col in dataTrain.columns if 'image_feature' in col]
    np.random.shuffle(imageFeat)
    dataTrain.drop(imageFeat, axis=1, inplace=True)
    dataTest.drop(imageFeat, axis=1, inplace=True)
    gc.collect();
    print(f'Final process done!')

    # Summarize dataset
    featureCnt = len(dataTrain.keys()) 
    numSamples = len(dataTrain)
    print(f'Training dataset has {numSamples} samples, and {featureCnt} features')
    featureCnt = len(dataTest.keys())
    numSamples = len(dataTest)
    print(f'Testing dataset has {numSamples} samples, and {featureCnt} features')

    print(f'Start training...')
    testRatio = 0.15
    np.random.seed(rs_idx*2)
    # Split training data randomly using train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        dataTrain, target, 
        test_size = testRatio,
        random_state = rs_idx,
        shuffle = True
    )
    # Create lgb dataset
    lgb_train = lgb.Dataset(
        data = x_train, 
        label = y_train,
        categorical_feature = catFeatures
    )
    lgb_test = lgb.Dataset(
        data = x_test, 
        label = y_test,
        categorical_feature = catFeatures
    ) 
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round = 8000,
        valid_sets = lgb_test,
        early_stopping_rounds = 400,
        verbose_eval = 500,
    )
    predResult = gbm.predict(dataTest, num_iteration=gbm.best_iteration)
    featureImportance = gbm.feature_importance()
    rmse = gbm.best_score['valid_0']['rmse']
    ff = time.time()
    print(f'Training done, RMSE = {rmse}: {ff-ss:.2f} seconds')

    if not os.path.exists(f'gbmResults'):
        os.mkdir(f'gbmResults')

    if not os.path.exists(f'{dataPath}/gbmResults'):
        os.mkdir(f'{dataPath}/gbmResults')
    
    pred_test = pd.DataFrame()
    pred_test['pred'] =  predResult
    pred_test['truth'] = target_val
    pred_test.to_csv(f"{dataPath}/gbmResults/predResult_{vers}_rsidx{rs_idx}_rmse{rmse:.5f}.csv")

    ftrs_df = pd.DataFrame()
    ftrs_df['features'] = dataTrain.columns
    ftrs_df['importance'] = featureImportance
    ftrs_df.sort_values(by='importance', ascending=False, inplace=True)
    ftrs_df.to_csv(f"{dataPath}/gbmResults/featureImportance_{vers}_rsidx{rs_idx}_rmse{rmse:.5f}.csv")
    ftrs_df.to_csv(f"gbmResults/featureImportance_{vers}_rsidx{rs_idx}_rmse{rmse:.5f}.csv")

    with open(f'{dataPath}/gbmResults/lgb_params_{vers}_rsidx{rs_idx}_rmse{rmse:.5f}.json', 'w') as f:
        json.dump(params, f, indent=4, ensure_ascii=False)
        
    ff = time.time()
    print(f'Parameters saved! {ff-ss:.2f}')
