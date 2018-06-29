import pandas as pd
import numpy as np
import time
from geopy import geocoders
g = geocoders.GoogleV3()

dataPath = '../../../data/avito-demand-prediction'

dataTrain = pd.read_csv(f'{dataPath}/train.csv.zip', compression='zip', usecols=['item_id', 'region', 'city'])
dataTest = pd.read_csv(f'{dataPath}/test.csv.zip', compression='zip', usecols=['item_id', 'region', 'city'])

dataTrain["city_region"] = dataTrain.loc[:, ["city", "region"]].apply(lambda l: " ".join(l), axis=1)
dataTest["city_region"] = dataTest.loc[:, ["city", "region"]].apply(lambda l: " ".join(l), axis=1)

uniqueCrList = np.concatenate([dataTrain.city_region.unique(), dataTest.city_region.unique()])
cr_unique = pd.DataFrame(uniqueCrList, columns=['city_region'])
cr_unique['geocode'] = cr_unique.city_region.apply(lambda x: g.geocode(x, timeout=10))
cr_unique['address'] = cr_unique.geocode.apply(lambda x: x.address)
cr_unique['latitude'] = cr_unique.geocode.apply(lambda x: x.latitude)
cr_unique['longitude'] = cr_unique.geocode.apply(lambda x: x.longitude)
cr_unique.drop('geocode', axis=1, inplace=True)

cr_unique.to_feather(f'{dataPath}/city_region_geocode.feather')
