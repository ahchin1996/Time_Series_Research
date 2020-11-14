# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 22:19:16 2020

@author: cooke
"""
from datetime import datetime
import os
from math import sqrt
import pandas as pd
import numpy as np
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.layers import LSTM,Dropout,Dense

# 控制顯卡內核
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True
sess0 = tf.compat.v1.InteractiveSession(config=config)

def get_result(path,fd,fd_2):
    # fd_path = 'D:/Data/Stock/new_data/^DJI/^DJI_2000.csv'
    df = pd.read_csv(path,sep=',',header=0)
    date_array = pd.to_datetime(df['Date'] )
    print("Number of rows and columns:", df.shape)
    year = fd_2.strip(f'{fd}')
    year = int(year.strip('_.csv'))

    id = 0
    while date_array.iloc[id] < datetime(year, 11, 1, 0, 0):
        id +=1
    print(id)

    df.drop(['Date'] ,axis=1 ,inplace=True)
    df.head(5)

    training_set = df.iloc[:id,:]
    test_set = df.iloc[id:,:]
    training_set, test_set = np.array(training_set), np.array(test_set)

    #訓練資列統一做歸一後再拆分
    sc_train = MinMaxScaler(feature_range = (0, 1))

    training_set_scaled = sc_train.fit_transform(training_set)

    training_data = training_set_scaled[:,1:]
    training_label = training_set_scaled[:,0]

    #測試資料先拆分在做歸一化
    test_data = test_set[:,1:]
    test_label = test_set[:,0]

    #label先做reshape
    training_label = training_label.reshape(training_label.shape[0],1)
    test_label = test_label.reshape(test_label.shape[0],1)

    sc_test_data = MinMaxScaler(feature_range = (0, 1))
    test_data = sc_test_data.fit_transform(test_data)

    sc_test_label = MinMaxScaler(feature_range = (0, 1))
    test_label = sc_test_label.fit_transform(test_label)

    #reshape
    training_data = training_data.reshape((training_data.shape[0],training_data.shape[1],1))
    test_data = test_data.reshape((test_data.shape[0],test_data.shape[1],1))

    print(f"Train_data shape : {training_data.shape}\n"
          f"Train_label shape :{training_label.shape}\n"
          f"Test_data shape :{test_data.shape}\n"
          f"Test_label shape :{test_label.shape}")


    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(training_data.shape[1], training_data.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    model.summary()

    # fit network
    history = model.fit(training_data, training_label, epochs=100, batch_size=128, validation_data=(test_data, test_label), verbose=2,shuffle=False)
    testPredict = model.predict(test_data)

    # trainScore = sqrt(mean_squared_error(training_label, trainPredict))
    # print('Train RMSE: %.4f' % (trainScore))
    testPredict = sc_test_label.inverse_transform(testPredict)
    test_data = test_data.reshape((test_data.shape[0], test_data.shape[1],))
    test_data = sc_test_data.inverse_transform(test_data)
    test_label = sc_test_label.inverse_transform(test_label)

    mse = mean_squared_error(test_label, testPredict)
    print('Test MSE: %.4f' % (mse))
    testScore = sqrt(mean_squared_error(test_label, testPredict))
    print('Test RMSE: %.4f' % (testScore))

    n = len(test_label)
    testPredict = testPredict.reshape((n,1))
    mape = sum(np.abs((test_label - testPredict)/test_label))/n*100
    print('Test MAPE: %.4f' % (mape))

    return  testScore,mape,mse

def find_fd(path):
    result = pd.DataFrame()

    # 函數功能: 遞迴顯示指定路徑下的所有檔案及資料夾名稱
    for fd in os.listdir(path):
        full_path=os.path.join(path,fd)
        if os.path.isdir(full_path):
            print('Enter dir:',full_path)
            lst = []
            for fd_2 in os.listdir(full_path):
                full_path_2 = os.path.join(full_path, fd_2)
                if os.path.isdir(full_path_2):
                    continue
                else:
                    print('檔案:', full_path_2)
                    rmse, mape,mse = get_result(full_path_2, fd,fd_2)
                    # lst.append(np.format_float_positional(mse,precision = 5))
                    lst.append(np.format_float_positional(rmse, precision=5))
                    lst.append(np.format_float_positional(mape, precision=5))

            dict = {fd: lst}
            x = pd.DataFrame(dict)
            result = pd.concat([result, x], axis=1)
        else:
            print('檔案:', full_path)
    return result

import time
start_time = time.time()
path = 'D:/Time_Series_Research/new_data'
output = find_fd(path)
finish_time = time.time()
print('Total times : {:.3f}'.format(finish_time-start_time))

output.to_csv(os.path.join(path, 'output_result.csv'), index=0, header=1)
