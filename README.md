# code-ml_project

In this empirical work a deep learning approach for classification of intraday financial time series data
was presented using a LSTM neural network. To this end a data set of hourly open, high, low,
close price data of the S&P500 was used. Apart from the log-transformed market price data
additional well-known technical indicators were calculated and used as training and prediction
features. The sequence length for training and testing was 15 periods. Hence, the trained model attatched one of three labels (neutral, long, short) to every hour in the test set - based on the past 15 OHLC/indicator values. The time period considered for model training and validating was 01/2011 – 12/2017. The final model was tested from 01/2018 – 12/2018.

Unfortunatly the model did not perform well and also all tweaking and tuning did not really help. Since the model architecture has no real influence on the performance it all comes down to the training data set. I am experimenting with different labeling approaches and feature compositions. With a relative extrema approach used for labeling I achieved an accuracy of 55-60% but still no real breakthrough. I will probably have to label candlesticks myself by hand ;) 

In total this repository contains 4 files: 

# feature selection&prepro
* import of data set  
* generating labels for the training set
* feature selection
* splitting and exporting training, validation and testing data

# model_building
In this part the actual NN is put together. Initially this project was conducted with Tensorflow 2.0 Alpha. With the version update some mechanics changed and I had to remove the hparam tuning part due to incompatibility issues. 
Also the evaluation of the classifier is included in this part. 

# output&analysis
This part was mostly for generating a very simplistic trading history based on the raw signals generated by the trained model. 

# permutations
In oder to evaluate the significance of the model's trading performance 1000 random permutations (same signals different oder) of the generated signals are produced in this section. Subsequently their performance is measured against the actual signals. 
