#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

from src.utils import plot_validation_results, plot_scores


df = pd.read_csv('data/a2-life-expectancy-cleaned.csv')
df.head()
df_X = df.iloc[:,0:-1]
df_y = df.iloc[:,-1]


#########################################################################################
### Your code starts here ############################################################### 
labelencoder= preprocessing.LabelEncoder()
df_X['Status'] = labelencoder.fit_transform(df_X['Status'])
### Your code ends here #################################################################
#########################################################################################

# Convert dataframes to numpy arrays
X, y = df_X.to_numpy(), df_y.to_numpy()

# Split dataset in training and test data (20% test data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#########################################################################################
### Your code starts here ###############################################################
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
### Your code ends here #################################################################
#########################################################################################

num_folds = 5
param_choices = [1, 2, 3, 5]

X_train_folds = []
y_train_folds = []

#########################################################################################
### Your code starts here ###############################################################

# Hint: you can use np.array_split to split the sample indices into folds here
X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)
### Your code ends here #################################################################
#########################################################################################

def get_regressor_scores(regressor_name):
    param_to_scores = {}

    for param in param_choices: 
        ## We want to keep track of the training scores and validation scores
        rmse_train, rmse_valid = [], []
        
        for i in range(num_folds):
            X_train_fold, X_valid_fold = None, None
            y_train_fold, y_valid_fold = None, None

            #########################################################################################
            ### Your code starts here ###############################################################
            
            # Hint: consider the np.setdiff1d function to construct the training folds here (optional; 
            # other ways are fine too)
            X_valid_fold = X_train_folds[i]
            y_valid_fold = y_train_folds[i]
            
            X_train_fold = np.concatenate(np.delete(X_train_folds, i))
            y_train_fold = np.concatenate(np.delete(y_train_folds, i))
            ### Your code ends here #################################################################
            ######################################################################################### 

            ## Train all the regressors one-by-one and discuss the results
            if regressor_name == 'KNN':
                regressor = KNeighborsRegressor(n_neighbors=param).fit(X_train_fold, y_train_fold)
            elif regressor_name == 'DT':
                regressor = DecisionTreeRegressor(max_depth=param).fit(X_train_fold, y_train_fold)
            elif regressor_name == 'RF':
                regressor = RandomForestRegressor(max_depth=param).fit(X_train_fold, y_train_fold)
            else:
                regressor = GradientBoostingRegressor(max_depth=param).fit(X_train_fold, y_train_fold)          
            
            ## Predict labels for for training validation set
            y_pred_fold_train = regressor.predict(X_train_fold)
            y_pred_fold_valid = regressor.predict(X_valid_fold)
        
            ## Keep track of training and validation scores
            rmse_train.append(mean_squared_error(y_train_fold, y_pred_fold_train, squared=False))
            rmse_valid.append(mean_squared_error(y_valid_fold, y_pred_fold_valid, squared=False))
            
        ## Keep track of all num_folds scores for current param (for plotting)
        param_to_scores[param] = (rmse_train, rmse_valid)
        
        ## Print statement for some immediate feedback
        print('param = {}, RMSE (training) = {:.3f}, RMSE (validation) = {:.3f} (stdev: {:.3f})'.format(param, np.mean(rmse_train), np.mean(rmse_valid), np.std(rmse_valid)))

    plot_validation_results(param_to_scores, regressor_name)
    plot_scores(param_to_scores, regressor_name)


regressors = ['KNN', 'DT', 'RF', 'GB']
for name in regressors:
    get_regressor_scores(name)

