import pandas as pd
import numpy as np
import random
import time

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import datetime

import matplotlib.pyplot as plt
import seaborn as sns

#Read Data

def read(csv):
    return pd.read_csv(csv)


#Explore Data

def stats(df):
    print('Shape of the dataframe is', df.shape)
    return df.describe()

def convert_dt(df, column):
    df[column] = pd.to_datetime(df[column], infer_datetime_format=True)

def time_frame(df, dt_column):
    start_date = df[dt_column].min()
    end_date = df[dt_column].max()

    print('start date is', start_date, 'and end date is', end_date)

def top_10(df, column1, column2):
    df_int = df.groupby(df[column1]).count()
    sorted_ = crime_type.sort_values(by=[column2], ascending=False)
    top10 = sorted_.iloc[0:10]

    return top10

def bottom_10(df, column1, column2):
    df_int = df.groupby(df[column1]).count()
    sorted_ = crime_type.sort_values(by=[column2], ascending=True)
    bot10 = sorted_.iloc[0:10]

    return bot10


#Create Training and Testing Sets

def train_test(df, size, random_ = random.randint(0, 100)):
    train, test = train_test_split(df, test_size = size, random_state = random_)

    return train, test


#Pre-Process Data

def fill_all_missing(df):
    df.fillna(df.select_dtypes(include='number').median().iloc[0], inplace=True)

def fill_missing(df, columns):
    for column in columns:
        df[column] = df[column].fillna(df[column].median())

def normalize(train, test, columns):
    train_n = train.copy()
    test_n = test.copy()

    for column in columns:
        df_mean = train[column].mean()
        df_std = train[column].std()

        train_n[column] = train[column].apply(lambda x: (x - df_mean)/df_std)
        test_n[column] = test[column].apply(lambda x: (x - df_mean)/df_std)

    return train_n, test_n


#Generate Features

def onehot(df, cols):
    df_dums = pd.get_dummies(df, columns = cols)
    return df_dums

def discretize(df, column, bins, labels, right = True):
    df[column + '_labels'] = pd.cut(x=df[column], bins=bins, labels=labels, right=right)
    return df

def booltonum(df, column):
    assert df[column].dtype == 'bool'
    df[column] = df[column].astype(int)

#Build Classifiers

def linreg(df, features, outcome, size, random_ = random.randint(0,100)):
    for feature in features:
        assert df[feature].dtype == 'int64'
    assert df[outcome].dtype == 'int64'
    start_time = time.time()
    train, test = train_test(df, size, random_ = random_)
    lin_reg = LinearRegression()
    train_n = train.copy()
    test_n = test.copy()
    for feature in features:
        train_n_int, test_n_int = normalize(df, feature, train, test)
        train_n[feature] = train_n_int[feature]
        test_n[feature] = test_n_int[feature]

    x = train_n[features]
    lin_reg.fit(x, train_n[outcome])
    best_line = lin_reg.predict(test_n[[features]])

    print('This ML model took', time.time() - start_time, 'to train')
    print('The MAE is', metrics.mean_absolute_error(test[outcome], best_line))
    print('The MSE is', metrics.mean_squared_error(test[outcome], best_line))
    print('The R^2 value is', metrics.r2_score(test[outcome], best_line))

    return lin_reg

def grid_search(train_n, test_n, models, grid, outcome):
    # --- Grid Search Pseudocode --- # 
    # Move this into a function in your pipeline.py file! 
    results = {}
    # Begin timer 
    start = datetime.datetime.now()

    features = [col for col in train_n.columns if col != outcome]

    # Loop over models 
    for model_key in models.keys():
        
        # Loop over parameters 
        for params in grid[model_key]: 
            print("Training model:", model_key, "|", params)
            
            # Create model 
            model = models[model_key]
            model.set_params(**params)
            
            # Fit model on training set 
            model.fit(train_n[features], train_n[outcome])

            # Predict on testing set 
            predictions = model.predict(test_n[features])

            # Evaluate predictions 
            evaluation = evaluate(test_n[outcome], predictions)
            
            # Store results in your results data frame
            results[(model_key, str(params))] = evaluation

    # End timer
    stop = datetime.datetime.now()
    print("Time Elapsed:", stop - start)

    results_df = pd.DataFrame(results).T.reset_index()
    results_df.columns = ['Model', 'Parameters', 'Accuracy Score', 'R2 Score']

    return results_df


#Evaluate Classifiers

def evaluate(actual, predictions): #run precision, accuracy, f1, r2, rss from sklearn

    return (accuracy_score(actual, predictions), metrics.r2_score(actual, predictions))
