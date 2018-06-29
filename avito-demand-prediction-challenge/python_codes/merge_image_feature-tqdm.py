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
# Progress bar
from tqdm import tqdm

def loadData(fl):
    ddf = dd.read_hdf(fl, '/features', mode='r')
    pbar.update(len(fl))
    return ddf.compute(scheduler='single-threaded')

if __name__ == "__main__":
    args = sys.argv
    sourceImgDir = str(args[1])
    if len(args) == 3:
        nThread = int(args[2])
    else:
        nThread = cpu_count()

    # Specify path variables 
    dataRootPath = '../../../data/avito-demand-prediction/images'
    imagePath = f'{dataRootPath}/{sourceImgDir}'
    featurePath = f'{dataRootPath}/{sourceImgDir}_features'
    hdfPath = f'{dataRootPath}/hdf_data'
    if not os.path.exists(hdfPath):
        os.makedirs(hdfPath)

    # Control flags
    USE_HDF5_COMPRESSION_ARG = {
        'format': 'table',
        'complib': 'blosc:zstd', 
        'complevel': 9
    }
    VERBOSE = 0
    
    # Enable gc
    gc.enable()
    
    # Initialize HDF5 file
    HDF5FilePath = f'{hdfPath}/{sourceImgDir}.h5'
    if os.path.exists(HDF5FilePath):
        os.remove(HDF5FilePath)
    #saveHDF5File = pd.HDFStore(HDF5FilePath, **USE_HDF5_COMPRESSION_ARG)

    # Get file list to prepare for merge
    featureFileList = np.array(os.listdir(featurePath))[:200]
    nFiles = len(featureFileList)
    featureFileList = [f'{featurePath}/{fn}' for fn in featureFileList]
    featureFileList = np.array_split(featureFileList, nThread)

    # Submit parallel map job
    print(f'Submitting map job with {nThread} threads ...')
    pbar = tqdm(total=nFiles)
    with Pool(processes=nThread) as p:
        df = list(p.imap(loadData, featureFileList))

    print(f'Map job done, merging dataframes ...')
    df = pd.concat(df, sort=False).reset_index(drop=True)

    print(df.head())
    print(df.shape)

    print(f'Merging done, saving to HDF5 file at {HDF5FilePath} ...')
    #saveHDF5File['features'] = df
    #saveHDF5File.close()

    print(f'All done, EOF')
