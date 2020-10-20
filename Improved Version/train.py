import numpy as np

import preprocessing_functions as pf
import config
from sklearn.utils import shuffle


import warnings
warnings.simplefilter(action='ignore')

data = pf.load_data(config.PATH_TO_DATASET)
data = pf.impute_na(data)
data = pf.remove_nonvarying_feature(data, 'property_type')
data = pf.location_cleaning(data, 'location')

for var in config.CONT_VARS:
    data[var] = data[var].apply(pf.remove_sqft_text)

# log transform numerical variables
for var in config.CONT_VARS:
    data[var] = pf.log_transform(data, var)

for var in config.CATEGORICAL_VARS:
    data[var] = pf.remove_rare_labels(data, var, config.FREQUENT_LABELS[var])

for var in config.CATEGORICAL_VARS:
    data[var] = pf.encode_categorical(data, var, config.ENCODING_MAPPINGS[var])

FEATURES = list(data.columns)
scaler = pf.train_scaler(data[FEATURES],config.OUTPUT_SCALER_PATH)

data[FEATURES] = scaler.transform(data[FEATURES])
data = shuffle(data)

X_train, X_test, y_train, y_test = pf.split_train_test(data, config.TARGET)

pf.train_model(X_train, y_train, config.OUTPUT_MODEL_PATH)
print('finished_training')