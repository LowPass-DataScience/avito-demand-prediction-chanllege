import pandas as pd
import numpy as np
import os
import json

dataPath = '../../../data/avito-demand-prediction'

dataPath += '/Word_frequency'

def get_ranking(field):

    with open(f'{dataPath}/{field}_nonzero.json', 'r') as f:
        nonzero = json.load(f)

    with open(f'{dataPath}/{field}_zero.json', 'r') as f:
        zero = json.load(f)


    nz = pd.DataFrame()
    nz['word'] = nonzero.keys()
    nz['freq1'] = nz.apply(lambda x: 
                   np.float(nonzero[x.word]['freq']), axis=1)
    
    z = pd.DataFrame()
    z['word'] = zero.keys()
    z['freq0'] = z.apply(lambda x: 
                   np.float(zero[x.word]['freq']), axis=1)
    
    dfall = pd.merge(nz, z, how='outer', on='word').fillna(0)
    dfall['freq_diff'] = dfall.apply(lambda x: 
                   x.freq1 - x.freq0, axis=1)
    dfall.drop(['freq0', 'freq1'], axis=1, inplace=True)
    dfall.sort_values(by='freq_diff', ascending=False, inplace=True)

    if not os.path.exists('word_ranking'):
        os.mkdir('word_ranking')


    dfall.to_csv(f'word_ranking/{field}_ranking.csv', index=False, header=True)

if __name__ == "__main__":
    get_ranking('title')
    get_ranking('desc')
