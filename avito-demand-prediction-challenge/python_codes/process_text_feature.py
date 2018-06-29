import pandas as pd
import numpy as np
import os
import text_feature_engineer as tfe
import gc
from multiprocessing import Pool, cpu_count, Array
import time
gc.enable()

# Disable some warnings
import warnings
warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)

dataPath = '../../../data/avito-demand-prediction'
size = {'train': 1503424, 'test': 508438}

# compression parameters
USE_HDF5_COMPRESSION_ARG = {
    'format': 'table', 
    'complib': 'blosc:zstd', 
    'complevel': 9
}
def text_processing(args):
    basic_Raw, text_Raw = args

    # Add basic text features
    text_basic = tfe.addTextFeature(text_Raw)
    #print(f'text feature preprocssed for text_basic')
    del text_Raw
    gc.collect()
    
    # Add further text features
    text_final = tfe.further_features_for_text(text_basic)  
    #print(f'text features added for text_basic')
    del text_basic
    gc.collect()
    
    # Merge with basic data
    text_final.drop('user_id', axis=1, inplace=True)
    basic_text_combined = basic_Raw.merge(text_final, how='left', on='item_id')
    del basic_Raw
    del text_final
    gc.collect()

    return basic_text_combined

def multi_process_data(conf, n_thread = int(cpu_count())):
    # Load basic data
    with pd.HDFStore(f'{dataPath}/basicData.h5') as hdf:
        basic_Raw = hdf[f'/{conf}Raw']
        print(f'{conf}_basic_Raw loaded')
        hdf.close()
    
    # Load text data
    with pd.HDFStore(f'{dataPath}/textData.h5') as hdf:
        text_Raw = hdf[f'/{conf}Raw']
        print(f'{conf}_text_Raw loaded')
        hdf.close()

    print(f'{n_thread} threads')

    ss = time.time()
    basic_Raw_split = np.array_split(basic_Raw, n_thread)
    text_Raw_split = np.array_split(text_Raw, n_thread)

    arg_lst = []
    for i in range(n_thread):
        arg_lst.append((basic_Raw_split[i], text_Raw_split[i]))

    with Pool(processes=n_thread) as p:
        result = p.map(text_processing, arg_lst)

    basic_text_combined_all = pd.concat(list(result), ignore_index=True)

    ff = time.time()
    print(f'{ff-ss:.2f} seconds past!')

    # Save to hdf storage
    if f'{conf}_basic_text_data.h5' in os.listdir(f'{dataPath}'):
        os.remove(f'{dataPath}/{conf}_basic_text_data.h5')
        print(f'remove old {dataPath}/{conf}_basic_text_data.h5')
    
    with pd.HDFStore(f'{dataPath}/{conf}_basic_text_data.h5', **USE_HDF5_COMPRESSION_ARG) as hdf:
        hdf['data'] = basic_text_combined_all
        hdf.close()
    print(f'Data saved to {dataPath}/{conf}_basic_text_data.h5')

    del basic_text_combined_all
    gc.collect()

if __name__ == "__main__":
    multi_process_data('train')
    multi_process_data('test')
