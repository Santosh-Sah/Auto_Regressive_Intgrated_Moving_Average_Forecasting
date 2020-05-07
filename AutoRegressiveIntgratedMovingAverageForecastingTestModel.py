# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:52:22 2020

@author: Santosh Sah
"""

from AutoRegressiveIntgratedMovingAverageForecastingUtils import (readAutoRegressiveIntgratedMovingAverageForecastingXTest, 
                                                                  readAutoRegressiveIntgratedMovingAverageForecastingModel, 
                                                                  saveAutoRegressiveIntgratedMovingAverageForecastingPredictedValues, 
                                                                  readAutoRegressiveIntgratedMovingAverageForecastingXTrain,
                                                                  readAutoRegressiveIntgratedMovingAverageForecastingPredictedValues)

from AutoRegressiveIntgratedMovingAverageForecastingVisualization import visualizeAutoRegressiveIntgratedMovingAverageForecastingPredictedValues

"""
test the model on testing dataset
"""
def testAutoRegressiveIntgratedMovingAverageForecastingModel():
    
    #reading the training dataset
    X_train = readAutoRegressiveIntgratedMovingAverageForecastingXTrain()
    
    #reading testing set
    X_test = readAutoRegressiveIntgratedMovingAverageForecastingXTest()
    
    start = len(X_train)
    
    end = len(X_train) + len(X_test) - 1
    
    #reading model from pickle file
    autoRegressiveIntgratedMovingAverageForecastingModel = readAutoRegressiveIntgratedMovingAverageForecastingModel()
    
    #forecasting
    #Passing dynamic=False means that forecasts at each point are generated using the full history up to that point (all lagged values).
    #Passing typ='levels' predicts the levels of the original endogenous variables. 
    #If we'd used the default typ='linear' we would have seen linear predictions in terms of the differenced endogenous variables.
    predictedValues = autoRegressiveIntgratedMovingAverageForecastingModel.predict(start = start, end = end, dynamic = False, typ = "levels").rename("ARIMA(0, 1, 0) Prediction")
    
    #saving the foreasted values
    saveAutoRegressiveIntgratedMovingAverageForecastingPredictedValues(predictedValues)

def plotAutoRegressiveIntgratedMovingAverageForecastingPredictedValues():
    
    #reading testing set
    X_test = readAutoRegressiveIntgratedMovingAverageForecastingXTest()
    
    #reading predicted value
    predictedValues = readAutoRegressiveIntgratedMovingAverageForecastingPredictedValues()
    
    #visualizing the predicted values with training set and the testing set
    visualizeAutoRegressiveIntgratedMovingAverageForecastingPredictedValues(X_test, predictedValues)
    
    
if __name__ == "__main__":
    #testAutoRegressiveIntgratedMovingAverageForecastingModel()
    plotAutoRegressiveIntgratedMovingAverageForecastingPredictedValues()