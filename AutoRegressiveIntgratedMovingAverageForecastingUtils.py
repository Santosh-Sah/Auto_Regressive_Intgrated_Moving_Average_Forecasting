# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:50:57 2020

@author: Santosh Sah
"""
import pandas as pd
import pickle
from statsmodels.tsa.stattools import adfuller
"""
Import dataset and read specific column. Split the dataset in training and testing set.
"""
def importAutoRegressiveIntgratedMovingAverageForecastingDataset(autoRegressiveIntgratedMovingAverageForecastingDatasetFileName):
    
    autoRegressiveIntgratedMovingAverageForecastingDataset = pd.read_csv(autoRegressiveIntgratedMovingAverageForecastingDatasetFileName,
                                                                         index_col='Date',parse_dates=True)
    
    #the dataset is monthly dataset. Hence setting its frequency as monthly.
    autoRegressiveIntgratedMovingAverageForecastingDataset.index.freq = "MS"
    
    return autoRegressiveIntgratedMovingAverageForecastingDataset

#splitting dataset into training and testing set
def splitAutoRegressiveIntgratedMovingAverageForecastingDataset(autoRegressiveIntgratedMovingAverageForecastingDataset):
    
    #splitting the dataset into training and testing set.
    autoRegressiveIntgratedMovingAverageForecastingTrainingSet = autoRegressiveIntgratedMovingAverageForecastingDataset.iloc[:252]
    autoRegressiveIntgratedMovingAverageForecastingTestingSet = autoRegressiveIntgratedMovingAverageForecastingDataset.iloc[252:]
    
    return autoRegressiveIntgratedMovingAverageForecastingTrainingSet, autoRegressiveIntgratedMovingAverageForecastingTestingSet

#test dataset is stationary or non stationary
def agumentedDickeyFullerTest(series,title=''):
    
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")
        
"""
Save training and testing dataset
"""
def saveTrainingAndTestingDataset(X_train, X_test):
    
    #Write X_train in a picke file
    with open("X_train.pkl",'wb') as X_train_Pickle:
        pickle.dump(X_train, X_train_Pickle, protocol = 2)
    
    #Write X_test in a picke file
    with open("X_test.pkl",'wb') as X_test_Pickle:
        pickle.dump(X_test, X_test_Pickle, protocol = 2)

"""
read X_train from pickle file
"""
def readAutoRegressiveIntgratedMovingAverageForecastingXTrain():
    
    #load X_train
    with open("X_train.pkl","rb") as X_train_pickle:
        X_train = pickle.load(X_train_pickle)
    
    return X_train

"""
read X_test from pickle file
"""
def readAutoRegressiveIntgratedMovingAverageForecastingXTest():
    
    #load X_test
    with open("X_test.pkl","rb") as X_test_pickle:
        X_test = pickle.load(X_test_pickle)
    
    return X_test

"""
Save AutoRegressiveIntgratedMovingAverageForecasting as a pickle file.
"""
def saveAutoRegressiveIntgratedMovingAverageForecastingModel(autoRegressiveIntgratedMovingAverageForecastingModel):
    
    #Write AutoRegressiveIntgratedMovingAverageForecastingModel as a picke file
    with open("AutoRegressiveIntgratedMovingAverageForecastingModel.pkl",'wb') as autoRegressiveIntgratedMovingAverageForecastingModel_Pickle:
        pickle.dump(autoRegressiveIntgratedMovingAverageForecastingModel, autoRegressiveIntgratedMovingAverageForecastingModel_Pickle, protocol = 2)

"""
read AutoRegressiveIntgratedMovingAverageForecasting from pickle file
"""
def readAutoRegressiveIntgratedMovingAverageForecastingModel():
    
    #load AutoRegressiveIntgratedMovingAverageForecastingModel model
    with open("AutoRegressiveIntgratedMovingAverageForecastingModel.pkl","rb") as autoRegressiveIntgratedMovingAverageForecastingModel:
        autoRegressiveIntgratedMovingAverageForecastingModel = pickle.load(autoRegressiveIntgratedMovingAverageForecastingModel)
    
    return autoRegressiveIntgratedMovingAverageForecastingModel

"""
Save AutoRegressiveIntgratedMovingAverageForecasting as a pickle file.
"""
def saveAutoRegressiveIntgratedMovingAverageForecastingModelForFullDataset(autoRegressiveIntgratedMovingAverageForecastingModelForFullDataset):
    
    #Write AutoRegressiveIntgratedMovingAverageForecastingModelForFullDataset as a picke file
    with open("AutoRegressiveIntgratedMovingAverageForecastingModelForFullDataset.pkl",'wb') as autoRegressiveIntgratedMovingAverageForecastingModelForFullDataset_Pickle:
        pickle.dump(autoRegressiveIntgratedMovingAverageForecastingModelForFullDataset, autoRegressiveIntgratedMovingAverageForecastingModelForFullDataset_Pickle, protocol = 2)

"""
read AutoRegressiveIntgratedMovingAverageForecasting from pickle file
"""
def readAutoRegressiveIntgratedMovingAverageForecastingModelForFullDataset():
    
    #load AutoRegressiveIntgratedMovingAverageForecastingModelForFullDataset model
    with open("AutoRegressiveIntgratedMovingAverageForecastingModelForFullDataset.pkl","rb") as autoRegressiveIntgratedMovingAverageForecastingModelForFullDataset:
        autoRegressiveIntgratedMovingAverageForecastingModelForFullDataset = pickle.load(autoRegressiveIntgratedMovingAverageForecastingModelForFullDataset)
    
    return autoRegressiveIntgratedMovingAverageForecastingModelForFullDataset

"""
save AutoRegressiveIntgratedMovingAverageForecasting PredictedValues as a pickle file
"""

def saveAutoRegressiveIntgratedMovingAverageForecastingPredictedValues(autoRegressiveIntgratedMovingAverageForecastingPredictedValues):
    
    #Write AutoRegressiveIntgratedMovingAverageForecastingPredictedValues in a picke file
    with open("AutoRegressiveIntgratedMovingAverageForecastingPredictedValues.pkl",'wb') as autoRegressiveIntgratedMovingAverageForecastingPredictedValues_Pickle:
        pickle.dump(autoRegressiveIntgratedMovingAverageForecastingPredictedValues, autoRegressiveIntgratedMovingAverageForecastingPredictedValues_Pickle, protocol = 2)

"""
read AutoRegressiveIntgratedMovingAverageForecasting PredictedValues from pickle file
"""
def readAutoRegressiveIntgratedMovingAverageForecastingPredictedValues():
    
    #load AutoRegressiveIntgratedMovingAverageForecastingPredictedValues
    with open("AutoRegressiveIntgratedMovingAverageForecastingPredictedValues.pkl","rb") as autoRegressiveIntgratedMovingAverageForecastingPredictedValues_pickle:
        autoRegressiveIntgratedMovingAverageForecastingPredictedValues = pickle.load(autoRegressiveIntgratedMovingAverageForecastingPredictedValues_pickle)
    
    return autoRegressiveIntgratedMovingAverageForecastingPredictedValues

"""
save AutoRegressiveIntgratedMovingAverageForecasting ForecastedValues as a pickle file
"""

def saveAutoRegressiveIntgratedMovingAverageForecastingForecastedValues(autoRegressiveIntgratedMovingAverageForecastingForecastedValues):
    
    #Write AutoRegressiveIntgratedMovingAverageForecastingForecastedValues in a picke file
    with open("AutoRegressiveIntgratedMovingAverageForecastingForecastedValues.pkl",'wb') as autoRegressiveIntgratedMovingAverageForecastingForecastedValues_Pickle:
        pickle.dump(autoRegressiveIntgratedMovingAverageForecastingForecastedValues, autoRegressiveIntgratedMovingAverageForecastingForecastedValues_Pickle, protocol = 2)

"""
read AutoRegressiveIntgratedMovingAverageForecastingForecastedValues from pickle file
"""
def readAutoRegressiveIntgratedMovingAverageForecastingForecastedValues():
    
    #load AutoRegressiveIntgratedMovingAverageForecastingForecastedValues
    with open("AutoRegressiveIntgratedMovingAverageForecastingForecastedValues.pkl","rb") as autoRegressiveIntgratedMovingAverageForecastingForecastedValues_pickle:
        autoRegressiveIntgratedMovingAverageForecastingForecastedValues = pickle.load(autoRegressiveIntgratedMovingAverageForecastingForecastedValues_pickle)
    
    return autoRegressiveIntgratedMovingAverageForecastingForecastedValues


