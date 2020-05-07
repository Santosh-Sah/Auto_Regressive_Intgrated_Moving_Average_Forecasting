# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:41:20 2020

@author: Santosh Sah
"""
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

from AutoRegressiveIntgratedMovingAverageForecastingUtils import (readAutoRegressiveIntgratedMovingAverageForecastingXTest, 
                                                                  readAutoRegressiveIntgratedMovingAverageForecastingPredictedValues)

"""

calculating AutoRegressiveIntgratedMovingAverageForecasting metrics

"""
def testAutoRegressiveIntgratedMovingAverageForecastingMetrics():
    
    #reading testing set
    X_test = readAutoRegressiveIntgratedMovingAverageForecastingXTest()
    
    #reading predicted value
    predictedValues = readAutoRegressiveIntgratedMovingAverageForecastingPredictedValues()
    
    meanSquredError = mean_squared_error(X_test, predictedValues)
    
    meanAbsoluteError = mean_absolute_error(X_test, predictedValues)
    
    rootMeanSquaredError = np.sqrt(mean_squared_error(X_test, predictedValues))
    
    print(meanSquredError) #62026572.70104942
    
    print(meanAbsoluteError) #6440.76361221758
    
    print(rootMeanSquaredError) #7875.695061456444
    
    
    
if __name__ == "__main__":
    testAutoRegressiveIntgratedMovingAverageForecastingMetrics()