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

def read_all(categorical = False):
    if categorical:
        claims = pd.read_csv("Data/claims1_categorical.csv")
        for i in range(2, 6):
            claims = claims.append(pd.read_csv(f"Data/claims{i}_categorical.csv"))
    else:
        claims = pd.read_csv("Data/claims1.csv")
        for i in range(2, 6):
            claims = claims.append(pd.read_csv(f"Data/claims{i}.csv"))
    return claims


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

def detect_outlier(data, outliers):
    
    threshold=3
    mean_1 = np.mean(data)
    std_1 =np.std(data)
    
    
    for y in data:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers


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

def linreg(train_n, test_n, train, test, features, outcome, size = None, random_ = random.randint(0,100)):
    start_time = time.time()

    lin_reg = LinearRegression()

    x = np.array(train_n[features]).reshape(-1, 1)
    lin_reg.fit(x, train[outcome])
    best_line = lin_reg.predict(np.array(test_n[[features]]).reshape(-1, 1))

    print('This ML model took', time.time() - start_time, 'to train')
    print('The MAE is', metrics.mean_absolute_error(test[outcome], best_line))
    print('The MSE is', metrics.mean_squared_error(test[outcome], best_line))
    print('The R^2 value is', metrics.r2_score(test[outcome], best_line))

    return lin_reg, best_line

def grid_search(train_n, test_n, models, grid, outcome):
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
    results_df.columns = ['Model', 'Parameters', 'Accuracy Score', 'Precision Score',
    'Recall Score', 'F1 Score']

    return results_df


#Evaluate Classifiers

def evaluate(actual, predictions): #run accuracy, precision, recall, f1

    return (accuracy_score(actual, predictions), metrics.precision_score(actual, predictions), 
    metrics.recall_score(actual, predictions), metrics.f1_score(actual, predictions))
