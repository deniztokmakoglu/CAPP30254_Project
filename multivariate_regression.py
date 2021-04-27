import pandas as pd
import numpy as np
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def create_var_combinations(data_columns, n):
    combinations = itertools.combinations(data_columns, n)
    return list(combinations)

def multivariate_regression(train, test, norm_train, norm_test, cp, dv, data_columns, n):
    lst_indep_vars = create_var_combinations(data_columns, n)
    lst_metrics = []
    for tpl in lst_indep_vars:
        lst_metrics.append(Regression_Evaluation(list(tpl), dv, norm_train, norm_test, train, test))
    return lst_metrics 
        
def MultivarLinearRegression(indep_var, dep_var, normalized_train, 
                     normalized_test, 
                     nn_train, nn_test):
    #Getting the features
    train_features_variable = normalized_train.loc[:,indep_var]
    test_features_variable = normalized_test.loc[:,indep_var]
    train_targets_variable = nn_train.loc[:,dep_var]
    test_targets_variable = nn_test.loc[:,dep_var]
    #Creating the regression object
    regr = linear_model.LinearRegression()
    regr.fit(np.array(train_features_variable), train_targets_variable)
    #Predicting the regression
    predicted_line_variable = regr.predict(np.array(test_features_variable)) #normalized
    return (test_targets_variable, predicted_line_variable)

def Regression_Evaluation(indep_var, dep_var, normalized_train, 
                     normalized_test, 
                     nn_train, nn_test):
    target, predicted = MultivarLinearRegression(indep_var, dep_var, normalized_train, 
                     normalized_test, 
                     nn_train, nn_test)
    return [tuple(indep_var), r2_score(target, predicted),mean_squared_error(target, predicted),mean_absolute_error(target, predicted)]