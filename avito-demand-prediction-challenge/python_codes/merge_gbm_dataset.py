# Merge processed image features into a single dataframe 
# table and save as a compressed HDF5 file
# Raw data are scattered individual [image ID].h5 files
# each containing one row and 500+ columns of extracted
# image features
#
# Chen Chen

## Load packages
import pandas as pd 
import numpy as np
import os 
from os import scandir, listdir

# data IO
import dask as da
import dask.dataframe as dd

# Multi-threading
import multiprocessing as mp
from multiprocessing import Process, Pool, cpu_count, Array, JoinableQueue

# Garbage collection
import gc

# Error handling and argument io
import sys

def loadData(fl):
    df = pd.read_hdf(fl, '/features', mode='r')
    df.image = df.image.apply(lambda x: x.replace('.jpg', ''))
    return df

if __name__ == "__main__":
    args = sys.argv
    if len(args) == 2:
        nThread = int(args[1])
    else:
        nThread = cpu_count()

    # Specify path variables 
    sourceImgDir = 'train_image_features'
    dataRootPath = '/home/ec2-user/kaggle/nvme/avito-demand-prediction/data'
    imagePath = f'{dataRootPath}/{sourceImgDir}'
    featurePath = f'{dataRootPath}/{sourceImgDir}'
    resultPath = f'{dataRootPath}/result'
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)
    VERBOSE = 0
    # Enable gc
    gc.enable()

    ss = time.time()
    with pd.HDFStore(f'{dataPath}/basicData.h5') as hdf:
        y = hdf['trainTarget']
        print(f'Target data loaded')
        hdf.close()
    with pd.HDFStore(f'{dataPath}/basic_text_lsa_rs_data.h5') as hdf:
        dataTrain = hdf['train']
        print(f'Train data loaded')
        dataTest = hdf['test']
        print(f'Test data loaded')
        hdf.close()
    imageDf = pd.read_hdf(f'{dataPath}/imageData.h5', key='/trainRaw', mode='r')
    dataTrain = dataTrain.merge(imageDf, on='item_id', how='left')
    imageDf = pd.read_hdf(f'{dataPath}/imageData.h5', key='/testRaw', mode='r')
    dataTest = dataTest.merge(imageDf, on='item_id', how='left')
    ## Final processing
    # Remove invalid index columns
    dataTrain.drop(
        [col for col in dataTrain.columns if 'item_id' in col or 'user_id' in col], 
        axis = 1, 
        inplace = True
    )
    dataTest.drop(
        [col for col in dataTest.columns if 'item_id' in col or 'user_id' in col], 
        axis = 1, 
        inplace = True
    )
    # Temporarily drop activation_date
    # dataTrain.drop('activation_date', axis=1, inplace=True)
    # dataTest.drop('activation_date', axis=1, inplace=True)
    refDate = datetime(2017, 1, 1)
    dataTrain.activation_date = dataTrain.activation_date.apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - refDate).days)
    dataTest.activation_date = dataTest.activation_date.apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - refDate).days)
    print(f'Final process done!')
    # Summarize dataset
    featureCnt = len(dataTrain.keys()) 
    numSamples = len(dataTrain)
    print(f'Training dataset has {numSamples} samples, and {featureCnt} features')
    featureCnt = len(dataTest.keys())
    numSamples = len(dataTest)
    print(f'Testing dataset has {numSamples} samples, and {featureCnt} features')
    del imageDf
    gc.collect()
    ff = time.time()
    print(f'Time elapsed: {ff-ss:.2f}s')  

    ## Merge image
    # load image features
    print('Start loading image features...')  
    ss = time.time()
    imageDf = pd.read_feather(f'{resultPath}/train_image_features.feather')
    ff = time.time()
    print(f'Image loaded, shape = {imageDf.shape}, time used {ff-ss:.2f}s')
    # Merge image features
    ss = time.time()
    dataTrain = dataTrain.merge(imageDf, on='image', how='left')
    ff = time.time()
    print(f'Image data merged, shape = {dataTrain.shape}, time used {ff-ss:.2f}s')
    
    # Clean memory
    del imageDf
    gc.collect();
    
    # Save to feather
    ss = time.time()
    featherFile = f'{resultPath}/train.feather'
    dataTrain.to_feather(featherFile)
    ff = time.time()
    print(f'Data saved to {featherFile}, time used {ff-ss:.2f}s')
    
    print(f'All done, EOF')
