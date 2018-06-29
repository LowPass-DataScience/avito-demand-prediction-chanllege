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

    jobList = np.array(listdir(featurePath), dtype=str)
    nJobs = len(jobList)
    jobList = np.array([f'{featurePath}/{fn}' for fn in jobList])
    print(f'Process with {nThread} threads, total jobs {nJobs}')

    # Submit parallel map job
    print(f'Submitting map job with {nThread} threads ...')
    with Pool(processes=nThread) as p:
        df = p.map(loadData, jobList)

    print(f'Map job done, merging dataframes ...')
    df = pd.concat(df, sort=False).reset_index(drop=True)

    # Initialize HDF5 file
    DataFilePath = f'{resultPath}/{sourceImgDir}.feather'
    print(f'Merging done, saving to Feather file at {DataFilePath} ...')
    if os.path.exists(DataFilePath):
        os.remove(DataFilePath)
    df.to_feather(DataFilePath)

    print(f'All done, EOF')