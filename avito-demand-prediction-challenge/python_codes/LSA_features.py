import pandas as pd
import numpy as np 

import nltk
from nltk.corpus import stopwords
russian_stop = set(stopwords.words('russian'))

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class LSA_app():
  def __init__(self, conf='title', *args, **kwargs):
    self.vec = TfidfVectorizer(ngram_range=(1,2), stop_words=russian_stop, *args, **kwargs)
    self.conf = conf
    
  def get_tfidf(self, train_df, test_df):
    train_df[self.conf].fillna('NA', inplace=True)
    test_df[self.conf].fillna('NA', inplace=True)
    self.full_tfidf = self.vec.fit_transform(train_df[self.conf].values.tolist() + test_df[self.conf].values.tolist())
    
  def get_df_tfidf(self, df):
    return self.vec.transform(df[self.conf].values.tolist())
    
  def apply_svd(self, n=100, *args, **kwargs): 
    self.svd_obj = TruncatedSVD(n_components=n)
    self.svd_obj.fit(self.full_tfidf)
    
  def get_svd(self, df, tfidf):
    df_svd = pd.DataFrame(self.svd_obj.transform(tfidf))
    df_svd.columns = ['SVD_'+self.conf+'_'+str(i+1) for i in range(self.svd_obj.n_components)]
    return pd.concat([df, df_svd], axis=1)
