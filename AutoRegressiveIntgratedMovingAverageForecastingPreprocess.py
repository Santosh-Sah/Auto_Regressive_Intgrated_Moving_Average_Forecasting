# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:51:38 2020

@author: Santosh Sah
"""

from AutoRegressiveIntgratedMovingAverageForecastingUtils import (importAutoRegressiveIntgratedMovingAverageForecastingDataset, saveTrainingAndTestingDataset, 
                                                         splitAutoRegressiveIntgratedMovingAverageForecastingDataset)

def preprocess():
    
    autoRegressiveIntgratedMovingAverageForecastingDataset = importAutoRegressiveIntgratedMovingAverageForecastingDataset("TradeInventories.csv")
    
    X_train, X_test = splitAutoRegressiveIntgratedMovingAverageForecastingDataset(autoRegressiveIntgratedMovingAverageForecastingDataset)
    
    saveTrainingAndTestingDataset(X_train, X_test)
    

if __name__ == "__main__":
    preprocess()