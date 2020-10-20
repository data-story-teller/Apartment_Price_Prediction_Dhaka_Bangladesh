import preprocessing_functions as pf
import config
from sklearn.utils import shuffle
from math import sqrt
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score

import warnings

def pred(data):
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
    data[FEATURES] = pf.scale_features(data, FEATURES, config.OUTPUT_SCALER_PATH)
    data = shuffle(data)

    #split into train-test
    X_train, X_test, y_train, y_test = pf.split_train_test(data, config.TARGET)

    # make predictions
    y_pred = pf.prediction(X_test, config.OUTPUT_MODEL_PATH)
    return y_pred, y_test




data = pf.load_data(config.PATH_TO_DATASET)
y_pred, y_test = pred(data)


# determine mse and rmse
print('test mse: {}'.format(int(mean_squared_error(y_test, y_pred))))
print('test rmse: {}'.format(int(sqrt(mean_squared_error(y_test, y_pred)))))
print('test r2: {}'.format(r2_score(y_test, y_pred)))
print()