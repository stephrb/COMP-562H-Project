# Imports
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import numpy as np 
from datascience import *
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
import random

# Subtracts the mean from each feature and divides by the standard deviation for each feature
def standardize_data(arr):
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)

    return (arr - mean[np.newaxis, :]) / std[np.newaxis, :]

# Takes in an np.array and removes a random feature for a percentage of rows and sets missing values to np.nan
def remove_random_features(arr, percent=.10):
    num_rows_to_modify = int(percent * arr.shape[0])
    rand_rows = np.random.choice(np.arange(arr.shape[0]), num_rows_to_modify, replace=False)
    new_arr = arr.copy()
    for i in rand_rows:
        rand_index = np.random.randint(0, arr.shape[1])
        new_arr[i, rand_index] = np.nan
    
    return new_arr, num_rows_to_modify

# Runs a simulation of n trials with an imputer, data removing strategy, and how much data is missing and returns an array of MSE
def simulate(imputer, data, trials, remove_func=remove_random_features, percent_missing=.10):
    np.random.seed(42)
    
    res = np.zeros(trials)
    
    for i in range(trials):
        data_with_missing_features = remove_func(data, percent_missing)
        imputed_data, m = imputer.fit_transform(data_with_missing_features)
        
        res[i] = (np.square(data - imputed_data)) / m
    return res

# removes a percentage of random features with no row conditions
def remove_random_features_row_independent(arr, percent=.10):
    num_points_to_modify = int(percent * arr.shape[0]*arr.shape[1])
    idxs = [(i,j) for i in np.arange(arr.shape[0]) for j in np.arange(arr.shape[1])]
    new_arr = arr.copy()
    for i,j in random.sample(idxs, num_points_to_modify):
        new_arr[i, j] = np.nan
    
    return new_arr, num_points_to_modify