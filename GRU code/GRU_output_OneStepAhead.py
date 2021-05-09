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
from keras.layers import Dense,GRU
from tensorflow.keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from feature_list import *
import time

# 控制顯卡內核
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True
sess0 = tf.compat.v1.InteractiveSession(config=config)

stime = time.time()

def get_result(path,fd,fd_2):
    # fd_path = 'D:/Data/Stock/new_data/^DJI/^DJI_2000.csv'
    df_all = pd.read_csv(path,sep=',',header=0)
    date_array = pd.to_datetime(df_all['Date'] )
    print("Number of rows and columns:", df_all.shape)

    feature_list = chose_list_all()
    df = df_all[['Date', 'Close']]
    for i in range(0, len(feature_list)):
        df = pd.concat([df, df_all.iloc[:, feature_list[i]]], axis=1)

    year = fd_2.strip(f'{fd}')
    year = int(year.strip('_.csv'))

    split_no = 0
    while date_array.iloc[split_no] < datetime(year, 11, 1, 0, 0):
        split_no +=1
    print(split_no)

    df.drop(['Date'] ,axis=1 ,inplace=True)
    df.head(5)

    sc_df = MinMaxScaler(feature_range=(0, 1))
    df = sc_df.fit_transform(df)

    train_set = df[:split_no, :]
    test_set = df[split_no:, :]

    train_data = train_set[:, 1:]
    train_label = train_set[:, 0]
    test_data = test_set[:, 1:]
    test_label = test_set[:, 0]

    print(f"Train_data shape : {train_data.shape}\n"
          f"Train_label shape :{train_label.shape}\n"
          f"Test_data shape :{test_data.shape}\n"
          f"Test_label shape :{test_label.shape}")

    custom_early_stopping = EarlyStopping(
    monitor='loss',
    patience=30,
    min_delta=0.01,
    mode='auto'
    )


    model = Sequential()
    model.add(GRU(units=75, return_sequences=True, input_shape=(train_data.shape[1], 1)))
    model.add(GRU(units=30, return_sequences=True))
    model.add(GRU(units=30, return_sequences=True))
    model.add(GRU(units=30))
    model.add(Dense(units=1))

    opt = Adam(learning_rate=0.001)
    model.compile(optimizer= opt, loss='mean_squared_error', metrics=['accuracy'])

    model.summary()

    new_test_label = []
    all_length = len(df)
    n = split_no

    for i in range(0, all_length - split_no):
        print()
        print(f"No. {i+1} Model training!  Total number of Training times {all_length - split_no}!\n")
        print(fd_2 + '\n')
        # training model
        a = train_data
        b = train_label
        a = a.reshape((a.shape[0], a.shape[1], 1))
        b = b.reshape(b.shape[0], 1)
        print(a.shape, b.shape)
        model.fit(a,
                  b,
                  epochs=100,
                  batch_size=64,
                  verbose=2, shuffle=False,
                  callbacks=[custom_early_stopping])

        # predicting
        x = test_data[i, :]
        x = x.reshape(1, x.shape[0], 1)
        testPredict = model.predict(x)
        # new_test_label = np.concatenate([new_test_label,testPredict],axis = 0)
        new_test_label.append(testPredict)

        # add next data
        y = test_data[i, :]
        y = y.reshape(1, y.shape[0])
        train_data = np.concatenate([train_data, y], axis=0)
        z = test_label[i].reshape(1, )
        train_label = np.concatenate([train_label, z], axis=0)

    new_test_label = np.array(new_test_label)
    new_test_label = new_test_label.reshape(new_test_label.shape[0], 1)
    new_test_set = np.concatenate([test_data, new_test_label], axis=1)
    new_test_set = sc_df.inverse_transform(new_test_set)
    new_test_label = new_test_set[:, new_test_set.shape[1] - 1]

    test_set = sc_df.inverse_transform(test_set)
    test_label = test_set[:, 0]

    rmse = sqrt(mean_squared_error(test_label, new_test_label))
    print('Test RMSE: %.4f' % (rmse))

    mape = np.mean(np.abs((test_label - new_test_label) / test_label)) * 100
    print('Test MAPE: %.4f' % (mape))



    return  rmse,mape

def find_fd(path):
    result = pd.DataFrame()
    lst = []
    # 函數功能: 遞迴顯示指定路徑下的所有檔案及資料夾名稱
    for fd in os.listdir(path):
        full_path=os.path.join(path,fd)
        if os.path.isdir(full_path):
            print('Enter dir:',full_path)
            rmse_list = []
            mape_list = []
            for fd_2 in os.listdir(full_path):
                full_path_2 = os.path.join(full_path, fd_2)
                if os.path.isdir(full_path_2):
                    continue
                else:
                    print('檔案:', full_path_2)
                    mse, mape = get_result(full_path_2, fd, fd_2)
                    # lst.append(np.format_float_positional(mse,precision = 5))
                    rmse_list.append(np.format_float_positional(rmse, precision=5))
                    mape_list.append(np.format_float_positional(mape, precision=5))
            dict_rmse = {fd: rmse_list}
            dict_mape = {fd: mape_list}
            x = pd.DataFrame(dict_rmse)
            y = pd.DataFrame(dict_mape)
            result = pd.concat([result, x], axis=1)
            result = pd.concat([result, y], axis=1)

        else:
            print('檔案:', full_path)
    return result

import time
start_time = time.time()
path ='D:/Time_Series_Research/new_data/ALL_DATA'
output = find_fd(path)

output.to_csv(os.path.join(path, 'GRU_result_all_feature.csv'), index=0, header=1)
print()
print_time("program completed in", stime)

