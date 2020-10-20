import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

# accuracy measures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

from more_itertools import unique_everseen

import joblib


# Individual pre-processing and training functions
# ================================================

def load_data(df_path):
    # Function loads data for training
    return pd.read_csv(df_path)


def impute_na(df):
    # function replaces NA by value entered by user
    # or by string Missing (default behaviour)
    return df.dropna()


def remove_nonvarying_feature(df, var):
    return df.drop([var], axis=1)


def location_cleaning(df, var):
    df['zone'] = df.apply(lambda _: '', axis=1)
    df['area'] = df.apply(lambda _: '', axis=1)
    for index, row in df[var].iteritems():
        row = df[var][index].split(',')[:-1]
        row = list(unique_everseen([row.strip() for row in row if row]))
        if len(row) == 3:
            row = [row[0] + ' ' + row[1]]
        if len(row) > 1:
            df['zone'][index], df['area'][index] = row[0], row[1]

    df = df.drop(['location'], axis=1)
    return df


def remove_sqft_text(x):
    sqft = x.split(' ')
    sqft[0] = sqft[0].replace(',', '')
    return int((sqft[0]))


def log_transform(df, var):
    # apply logarithm transformation to variable
    return np.log(df[var])


def remove_rare_labels(df, var, frequent_labels):
    # groups labels that are not in the frequent list into the umbrella
    # group Rare
    return np.where(df[var].isin(frequent_labels), df[var], 'Rare')


def encode_categorical(df, var, mappings):
    # replaces strings by numbers using mappings dictionary
    return df[var].map(mappings)


def train_scaler(df, output_path):
    scaler = MinMaxScaler()
    scaler.fit(df)
    joblib.dump(scaler, output_path)
    return scaler

def scale_features(df, FEATURES, scaler):
    scaler = joblib.load(scaler) # with joblib probably
    return scaler.transform(df[FEATURES])


def split_train_test(df, target):
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    X_train, X_test = train_set.drop(target, axis=1), test_set.drop(target, axis=1)
    y_train, y_test = train_set[target], test_set[target]

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, output_path):
    # initialise the model
    # lin_model = Lasso(alpha=0.005, random_state=0)
    lin_model = LinearRegression()
    # train the model
    lin_model.fit(X_train, y_train)

    # save the model
    joblib.dump(lin_model, output_path)

    return None


def prediction(df, model):
    model = joblib.load(model)
    return model.predict(df)