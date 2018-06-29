import pandas as pd
import numpy as np
import os
from multiprocessing import Pool, cpu_count
import gc
gc.enable()

dataPath = '../../../data/avito-demand-prediction'

def merge_all(conf):
    image_df = pd.read_feather(f'{dataPath}/{conf}_jpg_features.feather')
    print(f'Image features loaded!')
    image_df.image = image_df.image.apply(lambda x: x.replace('.jpg', ''))
    print(f'Image features modified!')

    with pd.HDFStore(f'{dataPath}/imageData.h5') as hdf:
        image_index = hdf[f'{conf}Raw']
        print(f'Image dataRaw loaded!')
        hdf.close()

    main_df = pd.read_feather(f'{dataPath}/basic_text_lsa_rs_imageI_data_{conf}.feather')
    main_df = pd.merge(main_df, image_df, how='left', on='image')

    if f'{dataPath}/all_data_{conf}.feather' in os.listdir(f'{dataPath}'):
        os.remove(f'{dataPath}/all_data_{conf}.feather')
        print(f'Remove old {dataPath}/all_data_{conf}.feather')

    main_df.to_feather(f'{dataPath}/all_data_{conf}.feather')

if __name__ == "__main__":
    merge_all('train')
    merge_all('test')


