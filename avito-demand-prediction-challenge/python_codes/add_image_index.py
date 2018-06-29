import pandas as pd
import numpy as np
import os
import gc
import time
gc.enable()

# Disable some warnings
import warnings
warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)

dataPath = '../../../data/avito-demand-prediction'

# compression parameters
USE_HDF5_COMPRESSION_ARG = {
    'format': 'table', 
    'complib': 'blosc:zstd', 
    'complevel': 9
}

def add_image_index(conf):
    print(f'Working on {conf} tables...')
    df_text = pd.read_hdf(f'{dataPath}/basic_text_lsa_rs_data.h5', key=f'/{conf}', mode='r')
    print('Main table loaded')
    image_df = pd.read_hdf(f'{dataPath}/imageData.h5', key=f'/{conf}Raw', mode='r')
    print('Image index table loaded')
    image_df.drop('user_id', inplace=True, axis=1)
    df = pd.merge(df_text, image_df, how='left', on='item_id')
    print('New table done')
    del df_text, image_df
    gc.collect()

    if f'basic_text_lsa_rs_imageI_data_{conf}.feather' in os.listdir(f'{dataPath}'):
        os.remove(f'{dataPath}/basic_text_lsa_rs_imageI_data_{conf}.feather')
        print(f'Remove old {dataPath}/basic_text_lsa_rs_imageI_data_{conf}.feather')

    ss = time.time()
    print('Saving to feather')
    df.to_feather(f'{dataPath}/basic_text_lsa_rs_imageI_data_{conf}.feather')
    ff = time.time()
    print(f'{ff-ss:.2f} seconds elapsed!')
    del df
    gc.collect()

if __name__ == "__main__":
    add_image_index('train')
    add_image_index('test')
