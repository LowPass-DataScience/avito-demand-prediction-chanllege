import pandas as pd
import numpy as np
import os
import json
import text_feature_engineer as tfe
import gc
from multiprocessing import Pool, cpu_count, Array
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

def get_ranking_score(conf, tlrk, dsrk):
    with pd.HDFStore(f'{dataPath}/textData.h5') as hdf:
        textdf = hdf[f'/{conf}Raw']
        print(f'text loaded')
        hdf.close()
    
    textdf.drop('user_id', inplace=True, axis=1)

    n_thread = cpu_count()

    ss = time.time()
    text_split = np.array_split(textdf, n_thread)
    
    arg_lst = []
    for i in range(n_thread):
        arg_lst.append((text_split[i], tlrk, dsrk))

    with Pool(processes=n_thread) as p:
        result = p.map(tfe.word_ranking_scores, arg_lst)

    text_all = pd.concat(list(result), ignore_index=True)
    return text_all    

if __name__ == "__main__":
    title_rank_df = pd.read_csv('word_ranking/title_ranking.csv')
    tlrk = title_rank_df.set_index('word').to_dict()['freq_diff']
    
    desc_rank_df = pd.read_csv('word_ranking/desc_ranking.csv')
    dsrk = desc_rank_df.set_index('word').to_dict()['freq_diff']

    st = time.time()
    print('Get ranking scores for training set')
    train_rs = get_ranking_score('train', tlrk, dsrk)
    ft = time.time()
    print(f'{ft-st:.2f} seconds')

    st = time.time()
    print('Get ranking scores for test set')
    test_rs = get_ranking_score('test', tlrk, dsrk)
    ft = time.time()
    print(f'{ft-st:.2f} seconds')

    print(f'loading data...')
    with pd.HDFStore(f'{dataPath}/basic_text_lsa_data.h5') as hdf:
        train_df = hdf['train']
        test_df = hdf['test']
        hdf.close()


    print(f'merging data...')
    train_df = train_df.merge(train_rs, how='left', on='item_id')
    test_df = test_df.merge(test_rs, how='left', on='item_id')
    
    print(f'saving data...')
    ## Save to hdf storage
    if f'basic_text_lsa_rs_data.h5' in os.listdir(f'{dataPath}'):
        os.remove(f'{dataPath}/basic_text_lsa_rs_data.h5')
        print(f'remove old {dataPath}/basic_text_lsa_rs_data.h5')
        
    with pd.HDFStore(f'{dataPath}/basic_text_lsa_rs_data.h5', **USE_HDF5_COMPRESSION_ARG) as hdf:
        hdf['train'] = train_df
        hdf['test'] = test_df
        hdf.close()
    print(f'Data saved to {dataPath}/basic_text_lsa_rs_data.h5')
