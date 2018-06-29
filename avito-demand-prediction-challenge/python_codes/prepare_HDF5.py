import pandas as pd
import numpy as np
import string
import os
import json
import gc
from sklearn.preprocessing import LabelEncoder
gc.enable()

dataPath='../../../data/avito-demand-prediction'

USE_HDF5_COMPRESSION_ARG = {
    'format': 'table', 
    'complib': 'blosc:zstd', 
    'complevel': 9
}

# Remove any existing hdf5 storage file since it does not support clean overwrite
for f in os.listdir(f'{dataPath}'): 
    if '.h5' in f and 'active' not in f:
        os.remove(f'{dataPath}/{f}')
        print(f'{dataPath}/{f} removed')

basicDataStore = pd.HDFStore(f'{dataPath}/basicData.h5', **USE_HDF5_COMPRESSION_ARG)
textDataStore = pd.HDFStore(f'{dataPath}/textData.h5', **USE_HDF5_COMPRESSION_ARG)
imageDataStore = pd.HDFStore(f'{dataPath}/imageData.h5', **USE_HDF5_COMPRESSION_ARG)

## Load csv.zip data
train = pd.read_csv(os.path.join(dataPath, 'train.csv.zip'), compression='zip')
print('train.csv loaded')
test = pd.read_csv(os.path.join(dataPath, 'test.csv.zip'), compression='zip')
print('test.csv loaded')

## Apply label encoding
mapping_folder = 'Label_encoding_basic'
if not os.path.exists(mapping_folder):
    os.mkdir(mapping_folder)

def compressMainTable(df):
    for col in df:
        if df[col].dtype=='object' and df[col].nunique() < 3000 and col != 'activation_date':
            print(f'encoding {col}...')
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            le_mapping = dict(zip(le.classes_, map(int, le.transform(le.classes_))))

            with open(os.path.join(mapping_folder, col+'.json'), 'w') as f:
                json.dump(le_mapping, f, indent=4, ensure_ascii=False)

            df[col] = le.fit_transform(df[col].astype(str)).astype(np.int16)

    df.price = df.price.fillna(-1).astype(np.int64)
    df.item_seq_number = df.item_seq_number.astype(np.int32)
    df.activation_date = pd.to_datetime(df.activation_date)
    return df

tmpCat = pd.concat([train,test], sort=False)
tmpCat = compressMainTable(tmpCat)
train = tmpCat.iloc[:train.shape[0],:]
test = tmpCat.iloc[train.shape[0]:,]
target = train.deal_probability
train.drop('deal_probability', axis=1, inplace=True)
test.drop('deal_probability', axis=1, inplace=True)

## Separate and drop advanced features into a different data file
textFeatures = ['title', 'description']
imageFeatures = ['image', 'image_top_1']

textDataStore['trainRaw'] = train[['item_id', 'user_id'] + textFeatures]
imageDataStore['trainRaw'] = train[['item_id', 'user_id'] + imageFeatures]
textDataStore['testRaw'] = test[['item_id', 'user_id'] + textFeatures]
imageDataStore['testRaw'] = test[['item_id', 'user_id'] + imageFeatures]

train.drop(textFeatures+imageFeatures, axis=1, inplace=True)
test.drop(textFeatures+imageFeatures, axis=1, inplace=True)

basicDataStore['trainRaw'] = train
basicDataStore['testRaw'] = test
basicDataStore['trainTarget'] = target

# Close HDF5
basicDataStore.close()
textDataStore.close()
imageDataStore.close()

# Clean up
del tmpCat
gc.collect();
