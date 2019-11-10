#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 01:44:50 2019

@author: trenschsetter
"""


#%% Prepping

# TensorFlow and tf.keras
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Helper libraries
import numpy as np
import pandas as pd

print(tf.__version__)

#%% Loading Data
file_path_train = "picked_training_set.csv"
file_path_validation = "picked_validation_set.csv"
file_path_full_validation = "full_validation_set.csv"
file_path_predict = "sp500_2017transformed.csv"

log_dir=""


#Building numpy arrays for training and testing. Label column must be last column to the right
lags = 15
features_per_candle = 9#num of fetures/values for each timestep (OHLC hourly, OHLC daily distances plus indicators)
combined_features = features_per_candle*lags

feature_columns =[]
for i in range(lags*features_per_candle):
    feature_columns.append(i)

train_features = np.loadtxt(file_path_train, skiprows=1, delimiter=',', usecols=feature_columns)
train_labels = to_categorical(np.loadtxt(file_path_train, skiprows=1, dtype='int64', delimiter=',', usecols=combined_features),dtype='int64')
validation_features = np.loadtxt(file_path_validation, skiprows=1, delimiter=',', usecols=feature_columns)
validation_labels = to_categorical(np.loadtxt(file_path_validation, skiprows=1, dtype='int64', delimiter=',', usecols=combined_features),dtype='int64')
full_validation_features = np.loadtxt(file_path_full_validation, skiprows=1, delimiter=',', usecols=feature_columns)
full_validation_labels = to_categorical(np.loadtxt(file_path_full_validation, skiprows=1, dtype='int64', delimiter=',', usecols=combined_features),dtype='int64')
prediction_features = np.loadtxt(file_path_predict, skiprows=1, delimiter=',', usecols=feature_columns)

# Reshape data set for recurrent network architecture
train_featuresRNN = train_features.reshape(len(train_features),lags,features_per_candle) #reshape(datarows, lags/timesteps, num of values for each timestep)
validation_featuresRNN = validation_features.reshape(len(validation_features),lags,features_per_candle)
full_validation_featuresRNN = full_validation_features.reshape(len(full_validation_features),lags,features_per_candle)
full_validation_labels = to_categorical(np.loadtxt(file_path_full_validation, skiprows=1, dtype='int64', delimiter=',', usecols=combined_features),dtype='int64')
prediction_featuresRNN = prediction_features.reshape(len(prediction_features),lags,features_per_candle)


#%% Build the model funciton and analysis function

def train_test_model(log_dir,hparams):
    model = tf.keras.Sequential([
            tf.keras.layers.LSTM(hparams['num_units'], return_sequences=True, input_shape=(lags,features_per_candle)),
            tf.keras.layers.Dropout(hparams['dropout_rate']),
            tf.keras.layers.BatchNormalization(),
             
            tf.keras.layers.LSTM(hparams['num_units'], return_sequences=False),
            tf.keras.layers.Dropout(hparams['dropout_rate']),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(3, activation='softmax')
    ])
    # Compiling = specifying the learning technique of the model
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
    )
    # Training the model
    model.fit(train_featuresRNN,train_labels, epochs=hparams['epochs'], #batch_size=32,
              validation_data=(validation_featuresRNN,validation_labels),
#              validation_data=(full_validation_featuresRNN,full_validation_labels),
              callbacks=[tf.keras.callbacks.TensorBoard(log_dir+"/keras-{}".format(model_name)),
                         #tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)
                         ])
    
    scores = model.evaluate(validation_featuresRNN,validation_labels)
    model.save("{}.h5".format(model_name))
    return scores

#%% Call function - Run model - Single Call
hyperparameter={'num_units':100, 'dropout_rate': 0.05, 'epochs':10}
name = 'nice_try{}'.format(lags)
model_name = str("model-{}".format(name))
print(model_name)
# Define hyperparameters for single train model call
history = train_test_model(log_dir, hyperparameter) #train model single call

#%% Load and use saved model 

new_model = tf.keras.models.load_model("model_name+".h5")

z= new_model.predict(prediction_featuresRNN)

b = pd.DataFrame(z,columns=['short','long','neutral']).round(decimals=4)

def predicted_labels(row):
    '''Takas as input rows with generated probabilities for the three labels from each hour, returns label with highest probability '''
    row=list(row)
    if row.index(max(row)) == 0:
        signal = 'short' 
    elif row.index(max(row)) == 1:
        signal = 'long'
    else:
        signal = 'neutral'
    return signal

signal_list=[]
for i in range(len(b)):
    signal_list.append(predicted_labels(list(b.loc[i,['short','long','neutral']])))

b['signal'] = signal_list

print('shorts:',b[b.signal=='short'].shape)
print('longs:',b[b.signal=='long'].shape)
print('neutrals:',b[b.signal=='neutral'].shape)
#Save output
pd.DataFrame(b).to_csv("/Users/trenschsetter/Dropbox/#0 Master Thesis/Data/saved_models/signals.csv",index=False)
#%%
actual_labels = pd.read_csv("sp500_2017actual_labels.csv", sep=',')            
c = b
c['actual_labels']=actual_labels.label.map({0:'short',1:'long',2:'neutral'})   

##--------------------------
model_long = c[c.signal=='long']
true_long = model_long[model_long.signal==model_long.actual_labels]
false_long = model_long[model_long.signal!=model_long.actual_labels]
fatal_long = model_long[model_long.actual_labels=='short']

#plt.hist(true_positive_long.long)
#plt.hist(false_positive_long.long)

#print('true positive avg probability =',mean(true_positive_long.long))
#print('false positive avg probability =',mean(false_positive_long.long))
##--------------------------

model_short = c[c.signal=='short']
true_short = model_short[model_short.signal==model_short.actual_labels]
false_short = model_short[model_short.signal!=model_short.actual_labels]
fatal_short = model_short[model_short.actual_labels=='long']

model_neutral = c[c.signal=='neutral']
true_neutral = model_neutral[model_neutral.signal==model_neutral.actual_labels]
false_neutral = model_neutral[model_neutral.signal!=model_neutral.actual_labels]

##--------------------------
# For recall only 'short' and 'long' will be consiedered
# True positives are true longs and shorts, false 
##--------------------------

print('longs:', len(model_long))
print('true_longs:', len(true_long))
#print('false_longs:', len(false_long))
print('shorts:', len(model_short))
print('true_shorts:', len(true_short))
#print('false_shorts:', len(false_short))
print('neutrals:', len(model_neutral))
print('true_neutrals:', len(true_neutral))
#print('false_neutrals:', len(false_neutral))
print('')
print('Accuracy:', len(c[c.signal==c.actual_labels])/len(c))
print('Accuracy active trading:', len(true_long+true_short)/len(model_long+model_short))
print('Precision long is:', len(true_long)/len(true_long+fatal_long))
print('Precision short is:', len(true_short)/len(true_short+fatal_short))
print('Trade accuracy is:', len(true_short+true_long)/len(true_short+true_long+fatal_short+fatal_long))


