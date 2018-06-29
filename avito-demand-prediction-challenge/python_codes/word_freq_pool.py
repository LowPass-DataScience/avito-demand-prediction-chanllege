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

## Add new function to pandas Series class
def force_agg(self, func, axis=0, *args, **kwargs):
    """ Force the aggregate function to work on all items instead of try Series.apply()"""
    axis = self._get_axis_number(axis)
    result, how = self._aggregate(func, *args, **kwargs)
    if result is None:
        kwargs.pop('_axis', None)
        kwargs.pop('_level', None)
        result = func(self, *args, **kwargs)
    return result
pd.Series.force_agg = force_agg

## Combine lists of list to a big list
def combine_lst(item):
    """ Return a pandas Series object """
    out = []
    for i in item:
        out.extend(i)
    return pd.Series(out)

if __name__ == "__main__":
    ## Load data
    with pd.HDFStore(f'{dataPath}/basicData.h5') as hdf:
        target = hdf[f'/trainTarget']
        print(f'target loaded')
        hdf.close()
        
    with pd.HDFStore(f'{dataPath}/textData.h5') as hdf:
        text = hdf[f'/trainRaw']
        print(f'text loaded')
        hdf.close()
        
    text_target_combine = pd.concat([text, target], axis=1)
    
    # Time Check Point
    t1 = time.time()
    
    ## Apply parallel processing for feature engineering
    n_thread = cpu_count()
    print(f'{n_thread} threads')
    text_target_combine_split = np.array_split(text_target_combine, n_thread)
    
    df_lst = []
    for i in range(n_thread):
        df_lst.append(text_target_combine_split[i])
        
    with Pool(processes=n_thread) as p:
        result = p.map(tfe.word_stem_collect, df_lst)
    
    text_target_all = pd.concat(list(result), ignore_index=True)
    
    # Time Check Point
    t2  = time.time()
    print(f'Feature Engineer done! {t2-t1:.2f} seconds')
    
    ## Pandas series for stem words pools
    NonZero_title_ps = text_target_all[text_target_all.deal_probability != 0].Stem_title.force_agg(combine_lst)
    Zero_title_ps = text_target_all[text_target_all.deal_probability == 0].Stem_title.force_agg(combine_lst)
    NonZero_desc_ps = text_target_all[text_target_all.deal_probability != 0].Stem_desc.force_agg(combine_lst)
    Zero_desc_ps = text_target_all[text_target_all.deal_probability == 0].Stem_desc.force_agg(combine_lst)
    
    # Time Check Point
    t3 = time.time()
    print(f'Combine lists! {t3-t2:.2f} seconds')

    ## Words ranking dict
    NonZero_title_dict = dict(NonZero_title_ps.value_counts())
    Zero_title_dict = dict(Zero_title_ps.value_counts())
    NonZero_desc_dict = dict(NonZero_desc_ps.value_counts())
    Zero_desc_dict = dict(Zero_desc_ps.value_counts())
    
    ## Total words count
    NonZero_title_count = len(NonZero_title_ps)
    Zero_title_count = len(Zero_title_ps)
    NonZero_desc_count = len(NonZero_desc_ps)
    Zero_desc_count = len(Zero_desc_ps)

    # Time Check Point
    t4 = time.time()
    print(f'Ranking dict processed! {t4-t3:.2f} seconds')

    ## Words frequency
    def get_tf(in_dict, count):
        out_dict = {}
        for k in in_dict.keys():
            out_dict[k] = {}
            out_dict[k]['freq'] = str(in_dict[k]/count)
            out_dict[k]['count'] = str(in_dict[k])
        return out_dict

    NonZero_title_tf = get_tf(NonZero_title_dict, NonZero_title_count)
    Zero_title_tf = get_tf(Zero_title_dict, Zero_title_count)
    NonZero_desc_tf = get_tf(NonZero_desc_dict, NonZero_desc_count)
    Zero_desc_tf = get_tf(Zero_desc_dict, Zero_desc_count)

    folder_name = 'Word_frequency'
    if not os.path.exists(f'{dataPath}/{folder_name}'):
        os.mkdir(f'{dataPath}/{folder_name}')

    with open(os.path.join(dataPath, folder_name, 'title_nonzero.json'), 'w') as f:
        json.dump(NonZero_title_tf, f, indent=4, ensure_ascii=False)

    with open(os.path.join(dataPath, folder_name, 'title_zero.json'), 'w') as f:
        json.dump(Zero_title_tf, f, indent=4, ensure_ascii=False)

    with open(os.path.join(dataPath, folder_name, 'desc_nonzero.json'), 'w') as f:
        json.dump(NonZero_desc_tf, f, indent=4, ensure_ascii=False)

    with open(os.path.join(dataPath, folder_name, 'desc_zero.json'), 'w') as f:
        json.dump(Zero_desc_tf, f, indent=4, ensure_ascii=False)

    # Time Check Point
    t5 = time.time()
    print(f'Term Frequency processed! {t5-t4:.2f} seconds')
