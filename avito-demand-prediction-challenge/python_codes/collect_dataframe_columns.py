import pandas as pd
import numpy as np
import os
import sys
import argparse

dataPath = '../../../data/avito-demand-prediction/features'

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--df", help="Input data storage file", type=str)
parser.add_argument("-m", "--message", help="Message for this data storage", type=str, default="")
parser.add_argument("-k", "--key", help="Key in HDF5 storage", type=str)
args = parser.parse_args()

if not os.path.exists('dataframe_info'):
    os.mkdir('dataframe_info')

def show_feather_info():
    msg = args.message
    msg += '\n'
    msg += 'column_name'+' '*19+'dtype'+' '*15+'null_count\n'
    try:
        dataframe = pd.read_feather(f'{dataPath}/{args.df}')
        print(f'Data loaded')
    except:
        print(f'Failed to load {dataPath}/{args.df}')
        return msg
    for c in dataframe.columns:
        name = c
        dtype = str(dataframe[c].dtype)
        null_count = str(dataframe[c].isnull().sum())
        msg += f'{name:30}{dtype:20}{null_count:12}\n'
    return msg

if __name__ == "__main__":
    if '.feather' in args.df:
        filename = args.df.replace('.feather', '_feather')
        filename = filename.replace('/', '__')
        m = show_feather_info()
        with open(f'dataframe_info/{filename}', 'w') as f:
            f.write(m)
