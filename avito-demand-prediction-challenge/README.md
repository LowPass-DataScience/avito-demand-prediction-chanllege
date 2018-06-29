# Avito Demand Prediction Challenge Competition Workflow

# Table of contents
- [Raw Data](#rawdata)
- [Data Engineering](#dataengineering)
    - [Save data to HDF5 storage](#s1)
    - [Add basic text features](#s2)
    - [Add LSA features](#s3)
    - [Add word ranking scores](#s4)
    - [Add Image features](#s5)
    - [Add Supervised Features](#s6)
    - [Add Geo information](#s7)
    - [Add LSA for 'parameter' features](#s8)
    - [Add information from train/test_active and train/test_periods datasets](#s9)
- [GradientBoost](#gbm)

## Raw Data <a name="rawdata"></a>
All the data is available: https://www.kaggle.com/c/avito-demand-prediction/data.

## Data Engineering <a name="dataengineering"></a>
### Step 1. Save data to HDF5 storage <a name="s1"></a>
`python prepare_HDF5.py` (for train/test.csv)

`python prepare_active_HDF5.py` (for train/test_active.csv)

- This step will apply label encoding to raw features using functions in 'label_encoder.py'
- Label encoding mapping dictionaries will be saved to 'Label_encoding_basic/' and 'Label_encoding_basic_active/' folders
- Save dataframe into HDF5 storage
- Input data:
  - train.csv.zip
  - test.csv.zip
  - train_active.csv.zip
  - test_active.csv.zip
- Output data:
  - basicData.h5
    - trainRaw
    - testRaw
    - trainTarget
  - textData.h5
    - trainRaw
    - testRaw
  - imageData.h5
    - trainRaw
    - testRaw
  - train_active_basic_Data-*num*.h5 (**manually moved to** *train_active_h5/* **folder**)
    - Raw  
  - test_active_basic_Data-*num*.h5 (**manually moved to** *test_active_h5/* **folder**)
    - Raw  
    
### Step 2. Add basic text features <a name="s2"></a>
`python process_text_features.py`
- This will add text features, including word counts, average word length, etc
- Functions to be used are in 'text_feature_engineer.py'
- Input data:
  - basicData.h5
  - textData.h5
- Output data:
  - train_basic_text_data.h5
    - data
  - test_basic_text_data.h5
    - data

### Step 3. Add LSA features <a name="s3"></a>
`python add_lsa_features.py`
- This will add latent semantic analysis features on title and description data
- Functions are located in 'LSA_features.py'
- Input data:
  - train_basic_text_data.h5
  - test_basic_text_data.h5
- Output data:
  - basic_text_lsa_data.h5
    - train
    - test

### Step 4. Add word ranking scores <a name="s4"></a>
`python word_freq_pool.py`
- In order to collect word stem pool for all training data, and split into zero-deal and non-zero-deal groups
- Functions are in 'text_feature_engineering.py'
- Input data: 
  - basicData.h5
    - trainRaw
    - trainTarget
- Output data:
  - Word_frequency/
    - title_nonzero.json
    - title_zero.json
    - desc_nonzero.json
    - desc_zero.json
    
`python get_word_ranking.py`
- Get difference of term frequencies between non-zero-deal and zero-deal groups
- Saving sorted (word, freq_diff) into csv files in 'Word_ranking/' folders
- Input data
  - Word_frequency/
    - title_nonzero.json
    - title_zero.json
    - desc_nonzero.json
    - desc_zero.json
- Output data: 
  - word_ranking/
    - title_ranking.csv
    - desc_ranking.csv

`python add_text_ranking_features.py`
- For each text field, get the sum of freq_diff using the ranked dataframe
- Input data:
  - word_ranking/
    - title_ranking.csv
    - desc_ranking.csv
- Output data:
  - basic_text_lsa_rs_data.h5
    - train
    - test
    

### Step 5. Add Image features <a name="s5"></a>
#### Preprocess data
`python add_image_index.py`
- Add image index for each item
- Input data:
  - basic_text_lsa_rs_data.h5
    - trainRaw
    - testRaw
  - imageData.h5
    - trainRaw
    - testRaw
- Output data: 
  - basic_text_lsa_rs_imageI_data_train.feather
  - basic_text_lsa_rs_imageI_data_test.feather

#### Process data
`python process_image.py {target image folder} {nParts} {part}`
- The following steps extracts image feature vector and save into HDFS file on a per image basis.
- Process image features. Use `tensorflow` to extract image feature vector using pre-trained tensorflow ImageNet models. Due the the computational load, the functions takes three arguments to segment the data to allow parallel processing. The three input arguments are `{target image feature}`, `{nParts}`, `{part}`, respectively. `{target image feature}` specifies the target image folder containing image files for feature extraction, *e.g.* `train_jpg_0.zip`. `{nParts}` specifies the number of parts to break down the input image files for parallelization, and `{part}` sets the current part number to run, which is in the range of `[1, nParts]`.
- Input data:
  - train_jpg_{part}.zip, part from 0~4
  - test_jpg.zip
- Output data:
  - {image id}.h5
    
#### Merge data
`python merge_image.py`
- This code merges previous generated individual HDFS files into several `feather` files containing concatenated dataframes.
- Input data:
  - processed image files `{image id}.h5`
- Output data:
  - merged feather file `{target image folder name}-{nParts}x{part}.feather`
  
#### Merge back to main table
`python main_image_merging.py`
- Merge image features back to main table 
- Input data:
  - basic_text_lsa_rs_imageI_data_train.feather
  - basic_text_lsa_rs_imageI_data_test.feather
  - **_train_jpg_features.feather_**
  - **_test_jpg_features.feather_**
- Output data:
  - all_data_train.feather
  - all_data_test.feather
  
### Step 6. Add Supervised Features <a name="s6"></a>
`python main_table_feature_engineering.py`
- Score features based on the deal probability
  - main table groupby 'category'
  - compute stats of deal probabilities within each 'category': mean, std
  - treat these values as scores for a certain 'category'
  - merge these scores to the main table on 'category' (both train and test table)
- Input data:
  - basicData.h5
    - trainRaw
    - testRaw
    - trainTarget
  - all_data_train.feather
  - all_data_test.feather
- Output data:
  - all_data_train_v1.feather
  - all_data_test_v1.feather

### Step 7. Add Geo information <a name="s7"></a>
`python collect_geo_info.py`
- Collect city latitude/longitude information
- Input data:
  - train.csv.zip
  - test.csv.zip
- Output data:
  - city_region_geocode.feather
  
`python add_city_info.py`
- Add city latitude, longitude, population into main dataset
- Input data:
  - all_data_train_v2.feather
  - all_data_test_v2.feather
  - city_region_geocode.feather
  - city_population_wiki_v3.csv (https://www.kaggle.com/stecasasso/russian-city-population-from-wikipedia)
  - train.csv.zip
  - test.csv.zip
- Output data:
  - all_data_train_v3.feather
  - all_data_test_v3.feather
  
### Step 8. Add LSA for 'parameter' features <a name="s8"></a>
`python add_lsa_params_featurs.py`
- Similar as Step 3. 
- Apply latent semantic analysis for param_1, param_2, param_3 parameters.
- Input data:
  - train.csv.zip
  - test.csv.zip
  - all_data_train_v3.feather
  - all_data_test_v3.feather
- Output data:
  - all_data_train_v4.feather
  - all_data_test_v4.feather
  
### Step 9. Add information from train/test_active and train/test_periods datasets <a name="s9"></a>
`python active_hdf5_to_feather.py`
- Input data: 
  - train_active_basic_Data-*num*.h5
    - Raw
  - test_active_basic_Data-*num*.h5 
    - Raw  
  - basicData.h5
    - trainRaw
    - testRaw
- Output data:
  - all_active_data.feather
  
`python add_active_features.py`
- Input data:
  - all_active_data.feather
  - basicData.h5
    - trainRaw
    - testRaw
- Output data:
  - train_active_grouped_data.feather
  - test_active_grouped_data.feather
  
`python add_periods_features.py`
- Input data:
  - all_active_data.feather
  - periods_train.csv.zip
  - periods_test.csv.zip
  - basicData.h5
    - trainRaw
    - testRaw
- Output data:
  - train_periods_grouped_data.feather
  - test_periods_grouped_data.feather

`python merge_active_periods_features.py`
- Input data: 
  - all_data_train_v4.feather
  - all_data_test_v4.feather
  - train_active_grouped_data.feather
  - test_active_grouped_data.feather
  - train_periods_grouped_data.feather
  - test_periods_grouped_data.feather
- Output data:
  - all_data_train_v5.feather
  - all_data_test_v5.feather

## Gradient Boost method <a name="gbm"></a>
`python LightGBM.py {random state}`
- Apply gradient boost method and predict deal probability for test dataset.
