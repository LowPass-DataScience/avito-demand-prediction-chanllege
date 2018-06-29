import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
import pickle as pkl
import time
import gc
import os
from multiprocessing import Pool, cpu_count, JoinableQueue, Process
from datetime import datetime
import tqdm
import dask as da
import dask.dataframe as dd
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
gc.enable()

randSeed = 1
np.random.seed(randSeed)

dataPath = '../../../data/avito-demand-prediction'

# Load raw data
dataTrain = pd.read_feather(f'{dataPath}/all_data_train_v2.feather')
dataTest = pd.read_feather(f'{dataPath}/all_data_test_v2.feather')

crGC = pd.read_feather(f'{dataPath}/city_region_geocode.feather')
crGC.rename(index=str, columns={"city_region": "region_city"}, inplace=True)
crGC.drop('address', axis=1, inplace=True)

cityPop = pd.read_csv(f'{dataPath}/city_population_wiki_v3.csv')
cityPop.population = cityPop.population.astype(np.int32)

crRawTrain = pd.read_csv(f'{csvPath}/train.csv.zip', compression='zip', usecols=['item_id', 'region', 'city'])
crRawTest = pd.read_csv(f'{csvPath}/test.csv.zip', compression='zip', usecols=['item_id', 'region', 'city'])
crRawTrain['region_city'] = crRawTrain.loc[:, ["city", "region"]].apply(lambda l: " ".join(l), axis=1)
crRawTest['region_city'] = crRawTest.loc[:, ["city", "region"]].apply(lambda l: " ".join(l), axis=1)
crRawTrain = crRawTrain.merge(cityPop, on='city', how='left')
crRawTest = crRawTest.merge(cityPop, on='city', how='left')
crRawTrain.drop(['region', 'city'], axis=1, inplace=True)
crRawTest.drop(['region', 'city'], axis=1, inplace=True)

crTrain = crRawTrain.merge(crGC, on='region_city', how='left')
crTest = crRawTest.merge(crGC, on='region_city', how='left')

# Label encode region_city
tmpCat = pd.concat([crTrain, crTest], sort=False)
tmpCat['region_city'] = le.fit_transform(tmpCat['region_city'].astype(str))
crTrain = tmpCat.iloc[:crTrain.shape[0],:]
crTest = tmpCat.iloc[crTrain.shape[0]:,:]
del tmpCat

# Merge to main table
dataTrain = dataTrain.merge(crTrain, on='item_id', how='left')
dataTest = dataTest.merge(crTest, on='item_id', how='left')

# Label encode user_id
tmpCat = pd.concat([dataTrain, dataTest], sort=False)
tmpCat['user_id'] = le.fit_transform(tmpCat['user_id'].astype(str))
dataTrain = tmpCat.iloc[:dataTrain.shape[0],:]
dataTest = tmpCat.iloc[dataTrain.shape[0]:,:]
del tmpCat
gc.collect();

# Drop image and fill na
dataTrain.drop('image', axis=1, inplace=True)
dataTest.drop('image', axis=1, inplace=True)
imageFeatCol = [col for col in dataTrain.columns if 'image' in col] + [
    'PCENT_over_exposed',
    'PCENT_under_exposed',
    'PCENT_normal_exposed',
    'entropy_hist',
    'CNT_corners',
    'CNT_local_peaks',
    'blurriness',
    'mean_color_B',
    'mean_color_G',
    'mean_color_R',
    'img_width',
    'img_height',
    'aspect_ratio',
    'average_pixel_width',
    'colorfulness'
]

dataTrain[imageFeatCol] = dataTrain[imageFeatCol].fillna(-1)
dataTest[imageFeatCol] = dataTest[imageFeatCol].fillna(-1)

# Save data to feather
dataTrain.to_feather(f'{dataPath}/all_data_train_v3.feather')
dataTest.to_feather(f'{dataPath}/all_data_test_v3.feather')
