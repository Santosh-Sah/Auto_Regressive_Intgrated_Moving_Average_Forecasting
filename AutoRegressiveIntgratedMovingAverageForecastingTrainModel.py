# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:51:54 2020

@author: Santosh Sah
"""
from statsmodels.tsa.arima_model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.statespace.tools import diff

from AutoRegressiveIntgratedMovingAverageForecastingUtils import (saveAutoRegressiveIntgratedMovingAverageForecastingModel, 
                                                         readAutoRegressiveIntgratedMovingAverageForecastingXTrain, 
                                                         importAutoRegressiveIntgratedMovingAverageForecastingDataset, 
                                                         saveAutoRegressiveIntgratedMovingAverageForecastingModelForFullDataset,
                                                         agumentedDickeyFullerTest)

from AutoRegressiveIntgratedMovingAverageForecastingVisualization import visualizeACFPlot, visualizePACFPlot


"""
Train AutoRegressiveIntgratedMovingAverageForecasting model on training set
"""
def trainAutoRegressiveIntgratedMovingAverageForecastingModel():
    
    X_train = readAutoRegressiveIntgratedMovingAverageForecastingXTrain()
    
    X_train["Inventories"] = X_train["Inventories"].astype('float64')
    
    #training model on the training set
    autoRegressiveIntgratedMovingAverageForecastingModel = ARIMA(X_train["Inventories"], order = (0, 1, 0))
    
    autoRegressiveIntgratedMovingAverageForecastingModelFitResult = autoRegressiveIntgratedMovingAverageForecastingModel.fit()
    
    saveAutoRegressiveIntgratedMovingAverageForecastingModel(autoRegressiveIntgratedMovingAverageForecastingModelFitResult)
    
    print(autoRegressiveIntgratedMovingAverageForecastingModelFitResult.summary())
    
# =============================================================================
#                                  ARIMA Model Results
#     ==============================================================================
#     Dep. Variable:          D.Inventories   No. Observations:                  251
#     Model:                 ARIMA(0, 1, 0)   Log Likelihood               -2550.053
#     Method:                           css   S.D. of innovations           6251.869
#     Date:                Thu, 07 May 2020   AIC                           5104.106
#     Time:                        16:28:22   BIC                           5111.157
#     Sample:                    02-01-1997   HQIC                          5106.944
#                              - 12-01-2017
#     ==============================================================================
#                      coef    std err          z      P>|z|      [0.025      0.975]
#     ------------------------------------------------------------------------------
#     const       3197.5697    394.614      8.103      0.000    2424.140    3971.000
#     ==============================================================================
# =============================================================================
    
"""
Train AutoRegressiveIntgratedMovingAverageForecasting model on full dataset
"""
def trainAutoRegressiveIntgratedMovingAverageForecastingModelOnFullDataset():
    
    autoRegressiveIntgratedMovingAverageForecastingDataset = importAutoRegressiveIntgratedMovingAverageForecastingDataset("TradeInventories.csv")
    
    autoRegressiveIntgratedMovingAverageForecastingDataset["Inventories"] = autoRegressiveIntgratedMovingAverageForecastingDataset["Inventories"].astype('float64')
    
    #training model on the whole dataset
    autoRegressiveIntgratedMovingAverageForecastingModel = ARIMA(autoRegressiveIntgratedMovingAverageForecastingDataset["Inventories"], order = (0, 1, 0))
    
    autoRegressiveIntgratedMovingAverageForecastingModelFitResult = autoRegressiveIntgratedMovingAverageForecastingModel.fit()
    
    saveAutoRegressiveIntgratedMovingAverageForecastingModelForFullDataset(autoRegressiveIntgratedMovingAverageForecastingModelFitResult)
    
    print(autoRegressiveIntgratedMovingAverageForecastingModelFitResult.summary())
    
# =============================================================================
#                                  ARIMA Model Results
#     ==============================================================================
#     Dep. Variable:          D.Inventories   No. Observations:                  263
#     Model:                 ARIMA(0, 1, 0)   Log Likelihood               -2672.018
#     Method:                           css   S.D. of innovations           6253.067
#     Date:                Thu, 07 May 2020   AIC                           5348.037
#     Time:                        16:30:42   BIC                           5355.181
#     Sample:                    02-01-1997   HQIC                          5350.908
#                              - 12-01-2018
#     ==============================================================================
#                      coef    std err          z      P>|z|      [0.025      0.975]
#     ------------------------------------------------------------------------------
#     const       3258.3802    385.581      8.451      0.000    2502.656    4014.104
#     ==============================================================================
# =============================================================================

def testIsDatasetStationary():
    
    autoAutoRegressiveIntgratedMovingAverageForecastingDataset = importAutoRegressiveIntgratedMovingAverageForecastingDataset("TradeInventories.csv")
    
    #order of p,d,q is SARIMAX(0, 1, 0)
    #hence we take the first difference as d is 1 to check stationarity.
    autoAutoRegressiveIntgratedMovingAverageForecastingDataset["diff1"] = diff(autoAutoRegressiveIntgratedMovingAverageForecastingDataset["Inventories"],
                                                              k_diff = 1)
    
    agumentedDickeyFullerTest(autoAutoRegressiveIntgratedMovingAverageForecastingDataset["diff1"])
    
# =============================================================================
#     Augmented Dickey-Fuller Test:
#     ADF test statistic       -3.412249
#     p-value                   0.010548
#     # lags used               4.000000
#     # observations          258.000000
#     critical value (1%)      -3.455953
#     critical value (5%)      -2.872809
#     critical value (10%)     -2.572775
#     Strong evidence against the null hypothesis
#     Reject the null hypothesis
#     Data has no unit root and is stationary
# =============================================================================
    
def determineARIMAOrderOfPAndQ():
    
    autoRegressiveIntgratedMovingAverageForecastingDataset = importAutoRegressiveIntgratedMovingAverageForecastingDataset("TradeInventories.csv")
    
    autoArimaResult = auto_arima(autoRegressiveIntgratedMovingAverageForecastingDataset["Inventories"], seasonal = False, trace = True)
    
    print(autoArimaResult.summary()) #order SARIMAX(0, 1, 0) 
    
# =============================================================================
#     Fit ARIMA(2,1,2)x(0,0,0,0) [intercept=True]; AIC=5373.961, BIC=5395.394, Time=0.361 seconds
#     Fit ARIMA(0,1,0)x(0,0,0,0) [intercept=True]; AIC=5348.037, BIC=5355.181, Time=0.011 seconds
#     Fit ARIMA(1,1,0)x(0,0,0,0) [intercept=True]; AIC=5399.843, BIC=5410.560, Time=0.044 seconds
#     Fit ARIMA(0,1,1)x(0,0,0,0) [intercept=True]; AIC=5350.241, BIC=5360.957, Time=0.040 seconds
#     Fit ARIMA(0,1,0)x(0,0,0,0) [intercept=False]; AIC=5409.217, BIC=5412.789, Time=0.009 seconds
#     Fit ARIMA(1,1,1)x(0,0,0,0) [intercept=True]; AIC=5378.835, BIC=5393.124, Time=0.185 seconds
#     Total fit time: 0.667 seconds
#                                    SARIMAX Results
#     ==============================================================================
#     Dep. Variable:                      y   No. Observations:                  264
#     Model:               SARIMAX(0, 1, 0)   Log Likelihood               -2672.018
#     Date:                Thu, 07 May 2020   AIC                           5348.037
#     Time:                        14:12:47   BIC                           5355.181
#     Sample:                             0   HQIC                          5350.908
#                                     - 264
#     Covariance Type:                  opg
#     ==============================================================================
#                      coef    std err          z      P>|z|      [0.025      0.975]
#     ------------------------------------------------------------------------------
#     intercept   3258.3802    470.991      6.918      0.000    2335.255    4181.506
#     sigma2       3.91e+07   2.95e+06     13.250      0.000    3.33e+07    4.49e+07
#     ===================================================================================
#     Ljung-Box (Q):                      455.75   Jarque-Bera (JB):               100.74
#     Prob(Q):                              0.00   Prob(JB):                         0.00
#     Heteroskedasticity (H):               0.86   Skew:                            -1.15
#     Prob(H) (two-sided):                  0.48   Kurtosis:                         4.98
#     ===================================================================================
# =============================================================================
    
def plotACFPlot():
    
    autoRegressiveIntgratedMovingAverageForecastingDataset = importAutoRegressiveIntgratedMovingAverageForecastingDataset("TradeInventories.csv")
    visualizeACFPlot(autoRegressiveIntgratedMovingAverageForecastingDataset)

def plotPACFPlot():
    
    autoRegressiveIntgratedMovingAverageForecastingDataset = importAutoRegressiveIntgratedMovingAverageForecastingDataset("TradeInventories.csv")
    visualizePACFPlot(autoRegressiveIntgratedMovingAverageForecastingDataset)
        
if __name__ == "__main__":
    #testIsDatasetStationary()   
    #determineARIMAOrderOfPAndQ()
    #plotACFPlot()
    #plotPACFPlot()
    #trainAutoRegressiveIntgratedMovingAverageForecastingModel()
    trainAutoRegressiveIntgratedMovingAverageForecastingModelOnFullDataset()
