#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 18:33:14 2019

@author: trenschsetter
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from statistics import mean, stdev as std
from scipy import stats
#%% Permutations generator

def generate_trading_history(signal_column):
    '''Dataframe needs to be named 'df_trades' and needs to have a column 'price' with times series of prices
    all other columns contain 'neutral', 'short' or 'long' signal in every row. 
    NAME FROM ANY IF THE SIGNAL COLUMNS IS PASSED AS INPUT FOR THIS FUNCTION'''
    trading_history =[]
    current_position = ''
    current_price = 0.0
    new_position = ''
    new_price = 0.0
    for i in range(len(df_trades)): #PnL takes into account open prices as 'price' for entry/exit points
        row=list(df_trades.loc[i,['price',signal_column]]) #iterates over all rows
        if row[1]== current_position: #if signal does not change - no realized PnL occurs - '0' is recorded in history
            trading_history.append(0)
        else:
            new_price = row[0]
            new_position = row[1]
            
            if current_position=='short':
                PnL= current_price - new_price
            elif current_position=='long':
                PnL= new_price - current_price 
            else:
                PnL=0
            current_position = new_position
            current_price = new_price
            trading_history.append(PnL)
    return sum(trading_history)


name = 'signals'
df_trades = pd.DataFrame()
permutations = []

data_to_predict = pd.read_csv("*directory*/sp500rawOHLC.csv", sep=',') #resampled unknown dataset wo/ labels
df_predicted = pd.read_csv("*directory*/saved_models/{}.csv".format(name), sep=',') #labels predicted by model for unknown data

df_trades['price'] = data_to_predict.Open
                           
predicted_labels = df_predicted.signal
random_signals = list(predicted_labels)
df_trades['random_signals'] = random_signals

for i in range(1000):
    if i%50==0:
        print('Current step:',i)
    random.shuffle(random_signals)
    df_trades['random_signals'] = random_signals
    permutations.append(generate_trading_history('random_signals'))

print('Permutations generated:',len(permutations))
plt.hist(permutations,bins=30)


simulated_permutations = pd.read_csv("*directory*/permutations.csv", sep=',')
stats.shapiro(simulated_permutations.PnL)
                                     
#%%
num_bins = 20
mu = mean(simulated_permutations.PnL)
sigma = std(simulated_permutations.PnL)

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(simulated_permutations.PnL, num_bins, density=1)

# add a 'best fit' line
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
ax.axvline(x=776.12, ymax = 0.75, color='r', ls='--')
ax.plot(bins, y, '--',)
ax.set_xlabel('cumulative PnL')
ax.set_ylabel('Probability density')
ax.set_title('Histogram permutations: n=1000, μ=-32.61, σ=403.81')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
#plt.show()

