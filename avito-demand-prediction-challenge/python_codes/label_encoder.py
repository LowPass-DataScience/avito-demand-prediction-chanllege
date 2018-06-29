from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Utility function to transform all categorical fields using one hot ending
def oneHotEncoding(df):
    # Get list categorical features
    catFeatures = [col for col in df.columns if df[col].dtype == 'object']
    # Convert to one hot encoding
    ohe = pd.get_dummies(df, columns=catFeatures)
    return ohe

# Utility function to transform all categorical fields using label encoder
def labelEncoding(df, catList=None):
    for col in df.columns:
        if df[col].dtype == 'object':
            # is categorical feature
            df[col] = le.fit_transform(df[col].astype(str))
            # (Optional) append to list
            if catList != None:
                catList.append(col)
            if VERBOSE:
                print(f'Feature {col} encoded with LabelEncoder')
    if catList != None:
        return df, catList
    else:
        return df

# Utility function to encode categorical features using LabelEncoder
def encodeLabel(field):
    return le.fit_transform(field.astype(str))
