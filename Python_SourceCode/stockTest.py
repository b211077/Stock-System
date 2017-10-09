# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:04:00 2017

@author: sjp
"""

       
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.callbacks
import time
from pandas import read_csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
#import math
#from sklearn.metrics import mean_squared_error

def load_data():
    #data = read_csv('C:/Users/sjp/Stock_Forcast/stockinfo2.csv', usecols = [0,1,6,7])
    data = read_csv('C:/Users/sjp/Stock_Forcast/testdata.csv', usecols = [0,1,2,5])
    cName = list(set(data['cname']))
    cName = cName[::-1]
    clist = pd.DataFrame(cName)
    #len(cName)
    return data, cName, clist


def make_dataset(df):
    dataset = df.values
    dataset = dataset[:,2:4]
    dataset = dataset.astype('float32')
    
    # fix random seed for reproducibility
    np.random.seed(7)

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    #split into train and test sets
    train_size = int(len(dataset) * 0.80)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    
    return dataset, train, test, scaler
      
        

def data_preprocessing(timesteps, futures, train, test):
    
    # num of train, test samples
    train_samples = len(train) - timesteps - futures
    test_samples = len(test) - timesteps - futures

    # convert trainset to LSTM array
    trainX_list = [np.expand_dims(np.atleast_2d(train[i:timesteps+i,:]), axis=0) for i in range(train_samples)]
    trainX = np.concatenate(trainX_list, axis=0)

    # convert trainresult to LSTM array
    trainY_list = [np.atleast_2d(train[i+timesteps+(futures-1):i+timesteps+futures,0]) for i in range(train_samples)]
    trainY = np.concatenate(trainY_list, axis=0)

    # convert testset to LSTM array
    testX_list = [np.expand_dims(np.atleast_2d(test[i:timesteps+i,:]), axis=0) for i in range(test_samples)]
    testX = np.concatenate(testX_list, axis=0)

    # convert testresult to LSTM array
    testY_list = [np.atleast_2d(test[i+timesteps+(futures-1):i+timesteps+futures,0]) for i in range(test_samples)]
    testY = np.concatenate(testY_list, axis=0)
    
    return train_samples, test_samples, trainX, trainY, testX, testY
    


def set_model(timesteps, features):
    
    # trials = trainX.shape[0]
    # features = trainX.shape[2] 
    neurons = [256, 256, 32, 1]

    model = Sequential()
    model.add(LSTM(neurons[0], input_shape=(timesteps, features), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(neurons[1], input_shape=(timesteps, features), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(neurons[2], kernel_initializer="uniform", activation='relu'))        
    model.add(Dense(neurons[3], kernel_initializer="uniform", activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def train_model(model, trainX, trainY, testX):
    
    history = LossHistory()
    model.fit(trainX, trainY, epochs=100, batch_size=512, callbacks=[history])
    #validation_split=0.05

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    return trainPredict, testPredict
    


def data_rescaling(scaler, features, trainPredict, testPredict, trainY, testY):
    
    # put the predictions 1st column
    # inverse transform
    trainPredict_extended = np.zeros((len(trainPredict),features)) #features -> 아웃풋 갯수
    trainPredict_extended[:,0:1] = trainPredict
    trainPredict = scaler.inverse_transform(trainPredict_extended)[:,0:1]

    testPredict_extended = np.zeros((len(testPredict),features))
    testPredict_extended[:,0:1] = testPredict
    testPredict = scaler.inverse_transform(testPredict_extended)[:,0:1]

    trainY_extended = np.zeros((len(trainY),features))
    trainY_extended[:,0:1] = trainY
    trainY = scaler.inverse_transform(trainY_extended)[:,0:1]

    testY_extended = np.zeros((len(testY),features))
    testY_extended[:,0:1] = testY
    testY = scaler.inverse_transform(testY_extended)[:,0:1]

    return trainPredict, testPredict, trainY, testY



def make_plot(dataset, scaler, timesteps, futures, trainPredict, testPredict):
    # make plot from train predictions
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[timesteps-1+futures:len(trainPredict)+timesteps-1+futures, :] = trainPredict
    # make plot from train predictions
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(timesteps*2)+futures:len(dataset)-futures, :] = testPredict

    # make plot from dataset
    datasetPlot = scaler.inverse_transform(dataset)

    # plot baseline and predictions
    plt.figure(figsize=(7,4))
    plt.plot(datasetPlot[:,0], color="gray", label='Dataset')
    plt.plot(trainPredictPlot[:,0], color='red', label='Train')
    plt.plot(testPredictPlot[:,0], color='blue', label='Test')
    plt.title('Stock Prediction')
    plt.xlabel('Data')
    plt.ylabel('Price')
    plt.legend(loc='lower right')
    plt.show()
    
    
    
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        

def train_score(trainY, trainPredict, timesteps, futures, train_samples):
    #set rate accuracy
    train_rate1 = [1 if trainY[i+timesteps+futures:i+timesteps+1+futures,0]-trainY[i+timesteps:i+timesteps+1,0]>=0 else -1 for i in range(train_samples)]
    train_rate2 = [1 if trainPredict[i+timesteps+futures:i+timesteps+1+futures,0]-trainPredict[i+timesteps:i+timesteps+1,0]>=0 else -1 for i in range(train_samples)]
    train_rate = [1 if train_rate1[i]==train_rate2[i] else 0 for i in range(len(train_rate1))]
    train_count = sum(1 for i in range(len(train_rate)) if train_rate[i] == 1)
    train_acc = train_count/len(train_rate) * 100
    return train_acc


def test_score(testY, testPredict, timesteps, futures, test_samples):
    test_rate1 = [1 if testY[i+timesteps+futures:i+timesteps+1+futures,0]-testY[i+timesteps:i+timesteps+1,0]>=0 else -1 for i in range(test_samples)]
    test_rate2 = [1 if testPredict[i+timesteps+futures:i+timesteps+1+futures,0]-testPredict[i+timesteps:i+timesteps+1,0]>=0 else -1 for i in range(test_samples)]
    test_rate = [1 if test_rate1[i]==test_rate2[i] else 0 for i in range(len(test_rate1))]
    test_count = sum(1 for i in range(len(test_rate)) if test_rate[i] == 1)
    test_acc = test_count/len(test_rate) * 100
    
    return test_acc
 
def predict_result(testPredict):
    decision = ['매수' if testPredict[-1] - testPredict[-2] > 0 else '매도']
    
    return decision


"""
def predict_score(testPredict, testY):
    pct_testPredict = pd.DataFrame(data=testPredict[:,0])
    pct_testPredict = pct_testPredict.pct_change()
    pct_testPredict = pct_testPredict.values
    pct_testPredict = pct_testPredict[~np.isnan(pct_testPredict)]
    pct_testPredict = pd.DataFrame(pct_testPredict)
    testPredict_rate = pct_testPredict * 100
    
    pct_testY = pd.DataFrame(data=testY[:,0])
    pct_testY = pct_testY.pct_change()
    pct_testY = pct_testY.values
    pct_testY = pct_testY[~np.isnan(pct_testY)]
    pct_testY = pd.DataFrame(pct_testY)
    testY_rate = pct_testY * 100
    
    testPredict_rate = testPredict_rate.values
    testY_rate = testY_rate.values
    
    rate1 = []
    rate2 = []
    
    for i in testPredict_rate:
        if i >= 0 :
            rate1.append('up')
        elif i < 0 :
            rate1.append('down')
            
    for i in testY_rate:
        if i >= 0 :
            rate2.append('up')
        elif i < 0 :
            rate2.append('down')
          
    rate = [1 if rate1[i:i+1] == rate2[i:i+1] else 0 for i in range(len(rate1))]
    test_count = sum(1 for r in rate if r == 1)
    result = test_count / len(testPredict_rate) * 100
    
    return result;
"""


if __name__ == "__main__":
    
    score = []
    recommend = []
    result = pd.DataFrame(score)
    
    global_start_time = time.time()
    
    print('> Loading data... ')
    data, cName, clist = load_data()
    
    for company in cName:
        df = data.loc[data['cname'] == company]
        dataset, train, test, scaler = make_dataset(df)
        
        # [features] num of input
        features = 2
        # [timesteps] for days of historical data
        timesteps = 15
        # [futures] for days later
        futures = 5
        
        # data preprocessing for LSTM input
        train_samples, test_samples, trainX, trainY, testX, testY = data_preprocessing(timesteps, futures, train, test)
        
        # set up model
        model = set_model(timesteps, features)
        
        # Train
        print('> Start train... ')
        trainPredict, testPredict = train_model(model, trainX, trainY, testX)
        
        # make resultset for prediction
        trainPredict, testPredict, trainY, testY = data_rescaling(scaler, features, trainPredict, testPredict, trainY, testY)
        
        # Result
        make_plot(dataset, scaler, timesteps, futures, trainPredict, testPredict)
        
        train_acc = train_score(trainY, trainPredict, timesteps, futures, train_samples)
        test_acc = test_score(testY, testPredict, timesteps, futures, test_samples)
        #test_acc = predict_score(testPredict, testY)
        decision = predict_result(testPredict)
        print('Train Score : ', train_acc)
        print('Test Score : ', test_acc)
        print('Decision : ', decision)
        
        score.append(test_acc)
        recommend.append(decision)
        
    tmp = pd.DataFrame(score)
    tmp2 = pd.DataFrame(recommend)
    result = pd.concat([clist, tmp, tmp2], axis=1)
    #result = pd.concat([clist, tmp], axis=1)
    result.to_csv('C:/Users/sjp/Stock_Forcast/testScore.csv', index=False, header=False)
    print('Entire Training duration (s) : ', time.time() - global_start_time)
    
    
"""
할일 :
    평가지표 / 최적화
    MACD 테스트, 아웃풋 3개, CUDA, 실시간 처리, Quandl 사용
"""