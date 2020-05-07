# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:53:28 2020

@author: Santosh Sah
"""
import pylab
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def visualizeAutoRegressiveIntgratedMovingAverageForecastingPredictedValues(autoRegressiveIntgratedMovingAverageForecastingXTest, 
                                                                            autoRegressiveIntgratedMovingAverageForecastingPredictedValues):
    
    #plotting the predicted values, and testing set
    title = 'Real Manufacturing and Trade Inventories'
    
    ylabel='Chained 2012 Dollars'
    
    xlabel='' 

    ax = autoRegressiveIntgratedMovingAverageForecastingXTest['Inventories'].plot(legend=True,figsize=(12,6),title=title)
    
    autoRegressiveIntgratedMovingAverageForecastingPredictedValues.plot(legend=True)
    
    ax.autoscale(axis='x',tight=True)
    
    ax.set(xlabel=xlabel, ylabel=ylabel)
    
    pylab.savefig('PredeictedValues.png')

def visualizeAutoRegressiveIntgratedMovingAverageForecastingForecastedValues(autoRegressiveIntgratedMovingAverageForecastingDataset, 
                                                                             autoRegressiveIntgratedMovingAverageForecastingForecastedValues):
    
    #plotting the predicted values, and testing set
    title = 'Real Manufacturing and Trade Inventories'
    
    ylabel='Chained 2012 Dollars'
    
    xlabel='' 

    ax = autoRegressiveIntgratedMovingAverageForecastingDataset['Inventories'].plot(legend=True,figsize=(12,6),title=title)
    
    autoRegressiveIntgratedMovingAverageForecastingForecastedValues.plot(legend=True)
    
    ax.autoscale(axis='x',tight=True)
    
    ax.set(xlabel=xlabel, ylabel=ylabel)
    
    pylab.savefig('ForecastedValues.png')

def visualizeACFPlot(autoRegressiveIntgratedMovingAverageForecastingDataset):
    
    title = 'Autocorrelation: Real Manufacturing and Trade Inventories'
    lags = 40
    plot_acf(autoRegressiveIntgratedMovingAverageForecastingDataset['Inventories'],title=title,lags=lags)
    pylab.savefig('acf_plot.png')

def visualizePACFPlot(autoRegressiveIntgratedMovingAverageForecastingDataset):
    
    title = 'Autocorrelation: Real Manufacturing and Trade Inventories'
    lags = 40
    plot_pacf(autoRegressiveIntgratedMovingAverageForecastingDataset['Inventories'],title=title,lags=lags)
    pylab.savefig('pacf_plot.png')