#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 20:25:11 2019

@author: chifle
"""

#%% Read in raw data from CSV, Construct resampled main data frame + add columns

import pandas as pd
import talib
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import mpl_finance
import random

#----------------------------------------------------------------------------#
## Market Price Data (SP500) was obtained from histdata.com
## This section is optional if you have the entire data set in one csv file already
## Columns are expected to be as specified in list 'df_header'

years = [2011,2012,2013,2014,2015,2016,2017,2018]

df_header=['Date', 'Open', 'High', 'Low', 'Close', 'Price']
df = pd.DataFrame()

for year in years:
    new_data = pd.read_csv("*file_name*_{}.csv".format(year),sep=';', names=df_header,parse_dates=['Date'],index_col='Date')
    df = df.append(new_data, sort=False)

#----------------------------------------------------------------------------#

df_main = pd.DataFrame()
df_daily = pd.DataFrame()
df_shifted = pd.DataFrame()
df_training = pd.DataFrame()

# Resample, then remove any line with a NaN
df_main['Open'] = df.Open.resample('1H').first()
df_main['High'] = df.High.resample('1H').max()
df_main['Low'] = df.Low.resample('1H').min()
df_main['Close'] = df.Close.resample('1H').last()

#adding daily data
df_daily['Open'] = df.Open.resample('1D').first()
df_daily['High'] = df.High.resample('1D').max()
df_daily['Low'] = df.Low.resample('1D').min()
df_daily['Close'] = df.Close.resample('1D').last()
df_daily['Datum'] = df_daily.index-timedelta(days=-1)
df_daily.reset_index(inplace=True)
df_daily.set_index('Datum',inplace=True)
df_daily.drop(columns='Date', inplace=True)

df_main.dropna(inplace=True)

df_main['Hour'] = df_main.index.hour 
df_main['RSI'] = talib.RSI(df_main.Open)
df_main['ADX'] = talib.ADX(df_main.High,df_main.Low,df_main.Close)
df_main['ATR'] = talib.ATR(df_main.High,df_main.Low,df_main.Close)
df_main['bollinger_up'],df_main['SMA'],df_main['bollinger_low'] = talib.BBANDS(((df_main.High + df_main.Close + df_main.Low)/3).values,timeperiod=20)
df_main['macd'],df_main['macdsignal'],df_main['macdhist'] = talib.MACD(((df_main.High + df_main.Close + df_main.Low)/3).values)

raw = pd.DataFrame()
raw['Open']=df_main['Open']
raw['High']=df_main['High']
raw['Low']=df_main['Low']
raw['Close']=df_main['Close']

df_main.dropna(how='any', inplace=True)

##---------------------------Plot and show BBANDS values---------------------##
#df_main[['Open','bollinger_up','bollinger_mid','bollinger_low']].head(20)
#plt.plot(df_main[['Open','bollinger_up','SMA20','bollinger_low']].head(200))

##---------------------------Plot Open price Data---------------------##
#plt.plot(df_main.Open, linewidth=0.5)
#plt.title('S&P500 stock index data from 01/2011 to 12/2018')
#plt.xlabel('Date')

#----------------------------------------------------------------------------#

#%% Labeling approach 
# Label added column based on distance of Open to next up- or down fractal
# If difference surpasses threshold 'short'/'long' label is added, otherwise 'neutral' label is added

labels = []
last_label='neutral'
threshold = 7

# Adding fractal indicator columns
upfractals=[]
downfractals=[]

for i in range(len(df_main)-2):
    if df_main.High[i-2]< df_main.High[i] and \
    df_main.High[i-1]< df_main.High[i] and \
    df_main.High[i+2]< df_main.High[i] and \
    df_main.High[i+1]< df_main.High[i]:
        upfractals.append(df_main.High[i])
    else:
        upfractals.append(0)

for i in range(len(df_main)-2):
    if df_main.Low[i-2]> df_main.Low[i] and \
    df_main.Low[i-1]>df_main.Low[i] and \
    df_main.Low[i+2] > df_main.Low[i] and \
    df_main.Low[i+1] > df_main.Low[i]:
        downfractals.append(df_main.Low[i])
    else:
        downfractals.append(0)

upfractals.append(0)
upfractals.append(0)
downfractals.append(0)
downfractals.append(0)

df_main['upfractal']=upfractals
df_main['downfractal']=downfractals

for index in range(len(df_main)):
    for index_upfractal in range(index,len(df_main)):
        if df_main.upfractal[index_upfractal]>0: # = if there is a value in the upfractal column --> check for next downfractal
            for index_downfractal in range(index,len(df_main)):
                if df_main.downfractal[index_downfractal]>0: # found next downfractal! Now compare positions and act accordingly
                    if index_upfractal < index_downfractal and df_main.upfractal[index_upfractal]-df_main.Open[index]>=threshold:
                        labels.append('long')
                    elif index_upfractal > index_downfractal and df_main.downfractal[index_downfractal]>df_main.Low[index] and df_main.upfractal[index_upfractal]-df_main.Open[index]>=threshold:
                        labels.append('long')
                    elif index_upfractal > index_downfractal and df_main.Open[index]-df_main.downfractal[index_downfractal]>=threshold:
                        labels.append('short')
                    elif index_upfractal < index_downfractal and df_main.upfractal[index_upfractal]<df_main.High[index] and df_main.Open[index]-df_main.downfractal[index_downfractal]>=threshold:
                        labels.append('short')
                    else:
                        labels.append('neutral')
                    break
            break

for i in range(len(df_main)-len(labels)): #filling last few cells with neutral label
    labels.append('neutral')

df_main['label']=labels

print(df_main[df_main.label=='short'].shape)
print(df_main[df_main.label=='long'].shape)
print(df_main[df_main.label=='neutral'].shape)

#%% Plot candle stick graphs w/labels

ax1 = plt.subplot2grid((1,1),(0,0))
_=plt.xticks(rotation=45)

plotframe = df_main.head(500)
chartlabels = df_main.label
y_coords = plotframe.Open

instances = []

##----------------------------------------------------------##
#Adding text to chart every time label changes
previous_label=''
for i in range(len(plotframe)):
    y = y_coords[i]
    if chartlabels[i] != previous_label:
        instances.append(i)
        plt.text(i,y,chartlabels[i],fontsize=9)
        plt.scatter(i, y, marker='x', color='blue')
        previous_label = chartlabels[i]
    else:
        pass
    
mpl_finance.candlestick2_ochl(ax1,plotframe.Open,plotframe.Close,plotframe.High,plotframe.Low, width=0.8)
##----------------------------------------------------------##


#%% Construct lagged differences data set (shifting)
df_shifted = pd.DataFrame()

lag = 15
feature_list = ['Open', 'High', 'Low', 'Close', 'Hour','RSI','ADX','ATR','bollinger_up','SMA','bollinger_low','macd','macdsignal','macdhist'] 

def shifting(feature, bar_amount):
    '''adds series with shifted values'''
    for i in range(bar_amount):
        df_shifted[feature+" shift"+str(i+1)] = df_main[feature].shift(periods=i+1)
        
for feature in feature_list:
    df_shifted=df_shifted.append(shifting(feature,lag))

    
#%% Construct training data set with selected features
    
df_training = pd.DataFrame()

for i in range(lag):
    df_training['diffopen'+str(i+1)] = np.log(df_main.Open)-np.log(df_shifted['Open shift'+str(i+1)])
    df_training['diffhigh'+str(i+1)] = np.log(df_main.Open)-np.log(df_shifted['High shift'+str(i+1)])
    df_training['difflow'+str(i+1)] = np.log(df_main.Open)-np.log(df_shifted['Low shift'+str(i+1)])
    df_training['diffclose'+str(i+1)] = np.log(df_main.Open)-np.log(df_shifted['Close shift'+str(i+1)])
#    df_training['rsi'+str(i+1)] = df_shifted['RSI shift'+str(i+1)]
 #   df_training['Hour'+str(i+1)] = df_shifted['Hour shift'+str(i+1)]
#    df_training['SMA'+str(i+1)] = df_shifted['SMA shift'+str(i+1)]
    df_training['ADX'+str(i+1)] = df_shifted['ADX shift'+str(i+1)]
    df_training['ATR'+str(i+1)] = df_shifted['ATR shift'+str(i+1)]
    df_training['macd'+str(i+1)] = df_shifted['macd shift'+str(i+1)]
    df_training['macdsignal'+str(i+1)] = df_shifted['macdsignal shift'+str(i+1)]
    df_training['macdhist'+str(i+1)] = df_shifted['macdhist shift'+str(i+1)]
#    
##---------------------------------------------------------------------------##

df_training['label'] = df_main.label.map({'short':0, 'long':1,'neutral':2})
df_training.dropna(axis=0,how='any', inplace=True) #drop rows with missing values (shift) 

#--------------HELPFUL FOR DATA EXPLORATION ----------------#

#plt.plot(df_training["diffopen1"],linewidth=0.5)
#plt.title('S&P500 log-transformed, differenced')
#plt.ylabel('diff lag_1 open')
#plt.xlabel('Date')
##plt.hist(df_training["diffhigh1"])
#
#plt.hist([df_training["diffopen1"],
#          df_training["diffopen2"],
#          df_training["diffopen3"],
#          df_training["diffopen4"],
#          df_training["diffopen5"]],
#          color=['blue', 'orange', 'black', 'red', 'green'],label=['diffopen1', 'diffopen2', 'diffopen3','diffopen4','diffopen5'], bins=20)
#plt.legend()
#plt.xlabel('range of differences')
#plt.ylabel('num of occurence')
#plt.title('Histogram distribution of log differences')

#%% Instances when label changes

end_date_testing = '2016-12-31'
end_date_validation = '2017-12-31'


final_training_set = df_training[df_training.index<=end_date_testing]
help_set = df_training[df_training.index>end_date_testing]
final_validation_set = help_set[help_set.index<end_date_validation]

##---------------------------------------------------------------------------##
# This part is optional. If used only takes instances into consideration where label changes
# which is where the label changes. 'Holding periods would not be used for training

instances = []
latest_label = ''
sp_labels = final_training_set.label
picked_training_set = pd.DataFrame()

#for i in range(len(final_training_set)):
#    if str(sp_labels.iloc[i]) != latest_label:
#        instances.append(i) 
#        instances.append(i+1)
#        instances.append(i+2)
#        instances.append(i+3)
##        instances.append(i+4)
##        instances.append(i+5)
#        latest_label = str(sp_labels.iloc[i])
#    else:
#        pass
#
#instances2 = []
#latest_label = ''
#sp_labels = final_validation_set.label
#picked_test_set = pd.DataFrame()
#
#for i in range(len(final_validation_set)):
#    if str(sp_labels.iloc[i]) != latest_label:
#        instances2.append(i) 
#        instances2.append(i+1)
#        instances2.append(i+2)
#        instances2.append(i+3)
##        instances2.append(i+4)
##        instances2.append(i+5)
#        latest_label = str(sp_labels.iloc[i])
#    else:
#        pass
##---------------------------------------------------------------------------##

final_training_set=final_training_set.reset_index()
final_validation_set=final_validation_set.reset_index()


##---------------------------------------------------------------------------##
#See above, needs to be activated if selected instances are to be used for training

#final_training_set = final_training_set[final_training_set.index.isin(instances)]
#final_validation_set = final_validation_set[final_validation_set.index.isin(instances2)]
##---------------------------------------------------------------------------##


#Prediction period wo/ labels (only resampled raw OHLC data); needed to generate trading history in output analysis part
raw[['Open','High','Low','Close']][raw.index>end_date_validation].to_csv("sp500rawOHLC.csv")

   
# Create data set for prediction (wo/ labels) and write out to CSV
predict_set = df_training[df_training.index>end_date_validation]


#export a csv file with just the label column
predict_set.label.to_csv("sp500_2017actual_labels.csv",header=True)
predict_set.drop('label',axis='columns', inplace=True)


#export the csv file of the transformed data with everything BUT the label column
predict_set.to_csv("sp500_2017transformed.csv",index=False)
##-----------------------------------------------------

final_training_set.drop('Date',axis='columns', inplace=True)
final_validation_set.drop('Date',axis='columns', inplace=True)
final_validation_set.to_csv("full_validation_set.csv",index=False)

##-----------------------------------------------------
# In this part is is made sure that labels in training set are equally distributed 
# in order to avoid bias during training. Instances are shuffled and chosen randomly.
                            
long_training = list(final_training_set.index[final_training_set.label==1])
short_training = list(final_training_set.index[final_training_set.label==0])
neutral_training = list(final_training_set.index[final_training_set.label==2])
random.shuffle(long_training)
random.shuffle(short_training)
random.shuffle(neutral_training)

short_training = short_training[:len(neutral_training)]
short_training = short_training[:len(long_training)]
long_training = long_training[:len(short_training)]
neutral_training = neutral_training[:len(short_training)]

    
indeces_training_picked = long_training+short_training+neutral_training
random.shuffle(indeces_training_picked)

picked_training_set = final_training_set[final_training_set.index.isin(indeces_training_picked)]
picked_training_set.to_csv("picked_training_set.csv",index=False)

##-----------------------------------------------------
long_validation = list(final_validation_set.index[final_validation_set.label==1])
short_validation = list(final_validation_set.index[final_validation_set.label==0])
neutral_validation = list(final_validation_set.index[final_validation_set.label==2])
random.shuffle(long_validation)
random.shuffle(short_validation)
random.shuffle(neutral_validation)

short_validation = short_validation[:len(neutral_validation)]
short_validation = short_validation[:len(long_validation)]
long_validation = long_validation[:len(short_validation)]
neutral_validation = neutral_validation[:len(short_validation)]
    
indeces_validation_picked = long_validation+short_validation+neutral_validation
random.shuffle(indeces_validation_picked)

picked_validation_set = final_validation_set[final_validation_set.index.isin(indeces_validation_picked)]
picked_validation_set.to_csv("picked_validation_set.csv",index=False)

##-----------------------------------------------------

print('signal short train count:',picked_training_set[picked_training_set.label==0].shape)
print('signal long train count:',picked_training_set[picked_training_set.label==1].shape)
print('signal neutral train count:',picked_training_set[picked_training_set.label==2].shape)
print('')
print('signal short validation count:',picked_validation_set[picked_validation_set.label==0].shape)
print('signal long validation count:',picked_validation_set[picked_validation_set.label==1].shape)
print('signal neutral validation count:',picked_validation_set[picked_validation_set.label==2].shape)
                             
                             
                           


            
