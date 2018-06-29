import pandas as pd
import numpy as np
import string
import os
import json
import gc
from sklearn.preprocessing import LabelEncoder
gc.enable()

dataPath='../../../data/avito-demand-prediction'

## Load csv.zip data
train_active = pd.read_csv(os.path.join(dataPath, 'train_active.csv.zip'), compression='zip')
print('train_active.csv loaded')
test_active = pd.read_csv(os.path.join(dataPath, 'test_active.csv.zip'), compression='zip')
print('test_active.csv loaded')

## Apply label encoding
mapping_folder = 'Label_encoding_basic_active'
if not os.path.exists(mapping_folder):
    os.mkdir(mapping_folder)

def compressMainTable(df):
    for col in df:
        if df[col].dtype=='object' and df[col].nunique() < 3000 and col != 'activation_date':
            print(f'encoding {col}...')
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            le_mapping = dict(zip(le.classes_, map(int, le.transform(le.classes_))))

            with open(os.path.join(mapping_folder, col+'.json'), 'w', encoding='utf-8') as f:
                json.dump(le_mapping, f, indent=4, ensure_ascii=False)

            df[col] = le.fit_transform(df[col].astype(str)).astype(np.int16)

    df.price = df.price.fillna(-1).astype(np.int64)
    df.activation_date = pd.to_datetime(df.activation_date)
    return df

tmpCat = pd.concat([train_active,test_active], sort=False)
tmpCat = compressMainTable(tmpCat)
train = tmpCat.iloc[:train_active.shape[0],:]
test = tmpCat.iloc[train_active.shape[0]:,]

## Store into hdf5 storage
# compression parameters
USE_HDF5_COMPRESSION_ARG = {
    'format': 'table', 
    'complib': 'blosc:zstd', 
    'complevel': 9
}

# Remove any existing hdf5 storage file since it does not support clean overwrite
for f in os.listdir(f'{dataPath}'): 
    if '.h5' in f and 'active' in f:
        os.remove(f'{dataPath}/{f}')
        print(f'{dataPath}/{f} removed')

# parameters for chunking
num_chunk_train = 15
num_chunk_test = 13
chunk_size = 1000000

# text Features storage
textFeatures = ['title', 'description']

# text features in train_active
flag = 0
for i in range(num_chunk_train):
    with pd.HDFStore(f'{dataPath}/train_active_text_Data_{i}.h5', **USE_HDF5_COMPRESSION_ARG) as active_hdf:
        active_hdf['Raw'] = train[['item_id', 'user_id'] + textFeatures][flag:flag+chunk_size]
        flag += chunk_size
        active_hdf.close()
      
# text features in test_active
flag = 0
for i in range(num_chunk_test):
    with pd.HDFStore(f'{dataPath}/test_active_text_Data_{i}.h5', **USE_HDF5_COMPRESSION_ARG) as active_hdf:
        active_hdf['Raw'] = test[['item_id', 'user_id'] + textFeatures][flag:flag+chunk_size]
        flag += chunk_size
        active_hdf.close()

# Drop text features
train.drop(textFeatures, axis=1, inplace=True)
test.drop(textFeatures, axis=1, inplace=True)

# basic features in train_active
flag = 0
for i in range(num_chunk_train):
    with pd.HDFStore(f'{dataPath}/train_active_basic_Data_{i}.h5', **USE_HDF5_COMPRESSION_ARG) as active_hdf:
        active_hdf['Raw'] = train[flag:flag+chunk_size]
        flag += chunk_size
        active_hdf.close()

# basic features in test_active
flag = 0
for i in range(num_chunk_test):
    with pd.HDFStore(f'{dataPath}/test_active_basic_Data_{i}.h5', **USE_HDF5_COMPRESSION_ARG) as active_hdf:
        active_hdf['Raw'] = test[flag:flag+chunk_size]
        flag += chunk_size
        active_hdf.close()


# Clean up
del tmpCat
gc.collect();
