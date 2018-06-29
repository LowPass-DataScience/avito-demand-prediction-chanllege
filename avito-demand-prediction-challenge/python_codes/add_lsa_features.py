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

# compression parameters
USE_HDF5_COMPRESSION_ARG = {
    'format': 'table', 
    'complib': 'blosc:zstd', 
    'complevel': 9
}

if __name__ == "__main__":
    ## Load preprocessed data
    with pd.HDFStore(f'{dataPath}/train_basic_text_data.h5') as hdf:
        train_basic = hdf[f'data']
        print(f'Train basic_text data loaded')
        hdf.close()
        
    with pd.HDFStore(f'{dataPath}/test_basic_text_data.h5') as hdf:
        test_basic = hdf[f'data']
        print(f'Test basic_text data loaded')
        hdf.close()

    ## Load text data
    with pd.HDFStore(f'{dataPath}/textData.h5') as hdf:
        train_text = hdf[f'/trainRaw']
        print(f'train text_Raw loaded')
        test_text = hdf[f'/testRaw']
        print(f'test text_Raw loaded')
        hdf.close()

    ## Apply LSA for title info
    st = time.time()
    title_lsa = lf.LSA_app(conf='title')
    title_lsa.get_tfidf(train_text, test_text)
    ft = time.time()
    print(f'finshed tfidf setup for title: {ft-st:.2f} seconds')

    st = time.time()
    title_lsa.apply_svd()
    ft = time.time()
    print(f'finshed svd setup for title: {ft-st:.2f} seconds')

    st = time.time()
    train_t_tfidf = title_lsa.get_df_tfidf(train_text)
    test_t_tfidf = title_lsa.get_df_tfidf(test_text)
    train_t_lsa_df = title_lsa.get_svd(train_text, train_t_tfidf)
    test_t_lsa_df = title_lsa.get_svd(test_text, test_t_tfidf)
    
    del train_t_tfidf
    del test_t_tfidf
    del title_lsa
    gc.collect()
    
    ft = time.time()
    print(f'finshed svd features for title: {ft-st:.2f} seconds')
    
    ## Apply LSA for description info
    st = time.time()
    desc_lsa = lf.LSA_app(conf='description')
    desc_lsa.get_tfidf(train_text, test_text)
    ft = time.time()
    print(f'finshed tfidf setup for desc: {ft-st:.2f} seconds')

    st = time.time()
    desc_lsa.apply_svd()
    ft = time.time()
    print(f'finshed svd setup for desc: {ft-st:.2f} seconds')

    st = time.time()
    train_d_tfidf = desc_lsa.get_df_tfidf(train_text)
    test_d_tfidf = desc_lsa.get_df_tfidf(test_text)
    train_lsa_df = desc_lsa.get_svd(train_t_lsa_df, train_d_tfidf)
    test_lsa_df = desc_lsa.get_svd(test_t_lsa_df, test_d_tfidf)
    
    del train_d_tfidf
    del test_d_tfidf
    del desc_lsa
    del train_t_lsa_df
    del test_t_lsa_df
    gc.collect()
    
    ft = time.time()
    print(f'finshed svd features for desc: {ft-st:.2f} seconds')
    
    ## Merge to basic dataframes
    train_lsa_df.drop(['user_id', 'title', 'description'], axis=1, inplace=True)
    test_lsa_df.drop(['user_id', 'title', 'description'], axis=1, inplace=True)
    train_lsa = train_basic.merge(train_lsa_df, how='left', on='item_id')
    test_lsa = test_basic.merge(test_lsa_df, how='left', on='item_id')
    
    del train_lsa_df
    del test_lsa_df
    del train_basic
    del test_basic
    gc.collect()
    
    ## Save to hdf storage
    if f'basic_text_lsa_data.h5' in os.listdir(f'{dataPath}'):
        os.remove(f'{dataPath}/basic_text_lsa_data.h5')
        print(f'remove old {dataPath}/basic_text_lsa_data.h5')
        
    with pd.HDFStore(f'{dataPath}/basic_text_lsa_data.h5', **USE_HDF5_COMPRESSION_ARG) as hdf:
        hdf['train'] = train_lsa
        hdf['test'] = test_lsa
        hdf.close()
    print(f'Data saved to {dataPath}/basic_text_lsa_data.h5')
