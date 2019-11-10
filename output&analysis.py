#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:53:23 2019

@author: trenschsetter
"""

import matplotlib.pyplot as plt
import pandas as pd
import random
from statistics import mean, stdev as std
import statsmodels.api as sm

#%%
name = 'signals'

def drawdown(equity):
    '''Returns the max drawdown from a list of cumulative PnL values'''
    high = 0
    currentDD = 0
    maxDD = 0
    for balance in equity:
        if balance > high:
            high = balance
        elif balance < high:
            currentDD = high - balance
            if currentDD > maxDD:
                maxDD = currentDD
        else:
            pass
    return maxDD


data_to_predict = pd.read_csv("sp500rawOHLC.csv", sep=',') #resampled unknown dataset wo/ labels
perfect_foresight = pd.read_csv("sp500_2017actual_labels.csv", sep=',')

df_predicted = pd.read_csv("{}.csv".format(name), sep=',') #labels predicted by model for unknown data

predicted_labels = df_predicted.signal


#% Plotting generated labels

#ax1 = plt.subplot2grid((1,1),(0,0))
#_=plt.xticks(rotation=45) 
#
#
#instances = []
#instances_predicted = []
#
##Adding text to chart every time label changes - ACTUAL LABELS
##previous_label=''
##for i in range(len(plotframe)):
##    y = y_coords[i]
##    if chartlabels[i] != previous_label:
##        instances.append(i)
##        plt.text(i,y-0.3,chartlabels[i],fontsize=6)
##        plt.scatter(i, y, marker='x', color='blue')
##        previous_label = chartlabels[i]
##    else:
##        pass
#    
##PREDICTED LABELS
#previous_label=''
#for i in range(len(data_to_predict)):
#    y = data_to_predict.Open[i]
#    if df_predicted.signal[i] != previous_label:
#        instances_predicted.append(i)
#        plt.text(i,y-0.8,df_predicted.signal[i],fontsize=6)
#        plt.scatter(i, y, marker='x', color='blue')
#        previous_label = df_predicted.signal[i]
#    else:
#        pass
#
#        
### Chart wo/ formatted x-axis
#mpl_finance.candlestick2_ochl(ax1,data_to_predict.Open,data_to_predict.Close,data_to_predict.High,data_to_predict.Low, width=0.8)


#% Trading history

df_trades = pd.DataFrame()
random_signals = []

df_trades['price'] = data_to_predict.Open
df_trades['trade'] = list(df_predicted.signal) 
df_trades['perfect_foresight'] = perfect_foresight.label.map({0: 'short',1:'long',2:'neutral'})  

for i in range(len(df_trades)):
    signal = random.randint(0,2)
    if signal==0:
        random_signals.append('short')
    elif signal==1:
        random_signals.append('long')
    else:
        random_signals.append('neutral')

df_trades['random_signals'] = random_signals


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
    return trading_history

trades_model=[]
trades_perfect_foresight =[]
trades_random=[]
trades_model = generate_trading_history('trade')
trades_perfect_foresight = generate_trading_history('perfect_foresight')
trades_random = generate_trading_history('random_signals')


def cumulated(trading_history):
    '''Returns cumulated PnL per hour'''
    newvalue=0.0
    balance = []
    for i in trading_history:
        newvalue = newvalue+i
        balance.append(newvalue)
    return balance

cumulated_PnL_model=[]
cumulated_PnL_perfect_foresight=[]
cumulated_PnL_random_signals=[]
cumulated_PnL_model = cumulated(trades_model)
cumulated_PnL_perfect_foresight=cumulated(trades_perfect_foresight)
cumulated_PnL_random_signals = cumulated(trades_random)

# Additional stuff
actual_trades_model =[] #PnL of every trade != 0 
actual_trades_sp_prices = []
actual_trades_perfect_foresight =[]
actual_trades_random = []
# equity balance over whole tested period
cumulated_cleaned = [] #equity balance wo/ no trading periods 

# Exclude periods with neutral label
for i in range(len(trades_model)):
    if trades_model[i]!=0:
        actual_trades_model.append(trades_model[i])
        actual_trades_sp_prices.append(data_to_predict.Open[i])

for i in range(len(actual_trades_perfect_foresight)):
    if trades_perfect_foresight[i]!=0:
        actual_trades_perfect_foresight.append(trades_perfect_foresight[i])

for i in range(len(actual_trades_random)):
    if trades_random[i]!=0:
        actual_trades_random.append(trades_random[i])

newvalue1=0.0

#Cumulated PnL per hour - cleaned trading history and for sp500 for beta calc
for i in actual_trades_model:
    newvalue1 = newvalue1+i
    cumulated_cleaned.append(newvalue1)
    
newvalue1=0.0

##-------------------------------------------------------------------------

profit_trades_model = list(filter(lambda x: x>0,actual_trades_model))
loss_trades_model = list(filter(lambda x: x<0,actual_trades_model))

print('num of trades:',len(actual_trades_model))
print("Profitable trades rate is",len(profit_trades_model)/len(actual_trades_model))
print("Average profit is",mean(profit_trades_model))
print("Average loss is",mean(loss_trades_model))
print('Mean PnL model:', mean(actual_trades_model).round(decimals=2),'per trade,',len(actual_trades_model),'trades in total')
print('Cum. PnL index points', cumulated_PnL_model[-1])
#print('alpha:',return_model-return_sp500)
print('Max DD model:', drawdown(cumulated_PnL_model))

#print('Mean PnL foresight:', mean(actual_trades_perfect_foresight).round(decimals=2),'per trade,',len(actual_trades_perfect_foresight),'trades in total' )
#print('Mean PnL random:', mean(actual_trades_random).round(decimals=2),'per trade,',len(actual_trades_random),'trades in total')

plt.plot(pd.Series(cumulated_PnL_model))
plt.plot(df_trades['price']-2679)#2244.50)#1274.25)#
plt.title('Cumulative PnL 2018 - selected subset lag15, SP500')
plt.ylabel('PnL index points')
plt.xlabel('Hours Feb2011-Dec 2018')
plt.legend(['model','S&P500'], loc='upper left')

#%% Calc beta 
#daily returns
data_to_predict['cumulative_model']= cumulated_PnL_model
data_to_predict['cumulative_model']=data_to_predict['cumulative_model']+2679
data_to_predict.reset_index(inplace=True)
daily_data=pd.DataFrame()

data_to_predict['Date'] = pd.to_datetime(data_to_predict['Date']) 
data_to_predict.set_index('Date',inplace=True)

daily_data['cumulative_model']= data_to_predict.cumulative_model.resample('1D').last()
daily_data['Close'] = data_to_predict.Close.resample('1D').last()
daily_data.dropna(inplace=True)
plt.plot(daily_data)

df_percent = pd.DataFrame()

df_percent['model'] = pd.Series(daily_data['cumulative_model'])
df_percent['sp500'] = pd.Series(daily_data['Close'])
df_percent['model']=df_percent['model'].pct_change()
df_percent['sp500'] = df_percent['sp500'].pct_change()
df_percent.dropna(inplace=True)

plt.scatter(df_percent.sp500,df_percent.model,s=7)
plt.xlabel('return S&P500 in pct')
plt.ylabel('return final model in pct')
plt.title('S&P500 and final model daily returns')

#make regression model 
model = sm.OLS(df_percent['model'], df_percent['sp500'])

#fit model and print results
results = model.fit()
print(results.summary())

#Sharpe Ratio
model_return= (daily_data['cumulative_model'].iloc[-1]-daily_data['cumulative_model'].iloc[0])/daily_data['cumulative_model'].iloc[0]
sharpe = (model_return-0.01)/std(df_percent['model'])
print('Sharpe Ratio is:', sharpe)

#%% Graphs

#Histograms
plt.hist(actual_trades_model,bins=70)
plt.xlabel('PnL closed trade')
plt.ylabel('Histogram')
plt.title('Realized PnL per hour - model')

plt.hist(trades_perfect_foresight,color='orange',bins=80) 
plt.title('Distribution PnL per trade - foresight')

##--------------------------------------------------------------------------###

plt.plot(cumulated_PnL_model)
plt.plot(cumulated_PnL_perfect_foresight)
plt.plot(cumulated_PnL_random_signals)
#plt.plot(data_to_predict.Open-2241)
plt.title('sp500 cumulated PnL')
plt.ylabel('PnL index points')
plt.xlabel('Hours Jan-Dec 2017')
plt.legend(['model','perfect_foresight','random_signals'], loc='upper left')

####------
plt.plot(pd.Series(cumulated_PnL_model)+2679)
plt.plot(df_trades['price'])
#plt.plot(data_to_predict.Open-2241)
plt.title('sp500 PnL 2017 in points')
plt.ylabel('PnL index points')
plt.xlabel('Hours Jan-Dec 2017')
plt.legend(['model','sp500 cumulative PnL'], loc='upper left')
####-----------

plt.hist(actual_trades_model,bins=50)

