# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:52:57 2020

@author: Santosh Sah
"""

from AutoRegressiveIntgratedMovingAverageForecastingUtils import (importAutoRegressiveIntgratedMovingAverageForecastingDataset, 
                                                                  saveAutoRegressiveIntgratedMovingAverageForecastingForecastedValues,
                                                                  readAutoRegressiveIntgratedMovingAverageForecastingForecastedValues, 
                                                                  readAutoRegressiveIntgratedMovingAverageForecastingModelForFullDataset)

from AutoRegressiveIntgratedMovingAverageForecastingVisualization import visualizeAutoRegressiveIntgratedMovingAverageForecastingForecastedValues

def forecastAutoRegressiveIntgratedMovingAverageForecastingModel():
    
    #reading the dataset
    autoRegressiveIntgratedMovingAverageForecastingDataset = importAutoRegressiveIntgratedMovingAverageForecastingDataset("TradeInventories.csv")
    
    #reading the model whichis trained on the whole dataset
    autoRegressiveIntgratedMovingAverageForecastingModel = readAutoRegressiveIntgratedMovingAverageForecastingModelForFullDataset()
    
    #forecasting for 11 months
    autoRegressiveIntgratedMovingAverageForecastingForecastedValues = autoRegressiveIntgratedMovingAverageForecastingModel.predict(len(autoRegressiveIntgratedMovingAverageForecastingDataset),
                                                                          len(autoRegressiveIntgratedMovingAverageForecastingDataset)+11,
                                                                          typ='levels').rename("ARIMA(0, 1, 0) Prediction")
    
    #saving the forecasted values
    saveAutoRegressiveIntgratedMovingAverageForecastingForecastedValues(autoRegressiveIntgratedMovingAverageForecastingForecastedValues)

def plotAutoRegressiveIntgratedMovingAverageForecastingForecastedValues():
    
    #reading the dataset
    autoRegressiveIntgratedMovingAverageForecastingDataset = importAutoRegressiveIntgratedMovingAverageForecastingDataset("TradeInventories.csv")
    
    #reading the forecated values
    autoRegressiveIntgratedMovingAverageForecastingForecastedValues = readAutoRegressiveIntgratedMovingAverageForecastingForecastedValues()
    
    #visualizing the forecated values
    visualizeAutoRegressiveIntgratedMovingAverageForecastingForecastedValues(autoRegressiveIntgratedMovingAverageForecastingDataset, 
                                                                             autoRegressiveIntgratedMovingAverageForecastingForecastedValues)

if __name__ == "__main__":
    #forecastAutoRegressiveIntgratedMovingAverageForecastingModel()
    plotAutoRegressiveIntgratedMovingAverageForecastingForecastedValues()
    