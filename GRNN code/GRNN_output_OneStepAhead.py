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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from neupy import algorithms
from feature_list import *

# hide INFO and WARNING message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# control the kernal inside of GPU
# assgin which one GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# automatic selection running device
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
# 指定GPU顯示卡記憶體用量上限
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
# 自動增長GPU記憶體用量
# config.gpu_options.allow_growth = True
sess0 = tf.compat.v1.InteractiveSession(config=config)

stime = time.time()

def get_result(path,fd,fd_2):
    # fd_path = 'D:/Data/Stock/new_data/^DJI/^DJI_2000.csv'
    df = pd.read_csv(path,sep=',',header=0)
    date_array = pd.to_datetime(df['Date'] )
    print("Number of rows and columns:", df.shape)
    year = fd_2.strip(f'{fd}')
    year = int(year.strip('_.csv'))

    #get feature list
    feature_list = chose_list_all()

    new_df = df[['Date', 'Close']]
    for i in range(0, len(feature_list)):
        new_df = pd.concat([new_df, df.iloc[:, feature_list[i]]], axis=1)

    split_no = 0
    while date_array.iloc[split_no] < datetime(year, 11, 1, 0, 0):
        split_no +=1
    print(split_no)

    new_df.drop(['Date'] ,axis=1 ,inplace=True)
    new_df.head(5)

    sc_df = MinMaxScaler(feature_range=(0, 1))
    new_df = sc_df.fit_transform(new_df)

    train_set = new_df[:split_no, :]
    test_set = new_df[split_no:, :]

    train_data = train_set[:, 1:]
    train_label = train_set[:, 0]
    test_data = test_set[:, 1:]
    test_label = test_set[:, 0]

    print(f"Train_data shape : {train_data.shape}\n"
          f"Train_label shape :{train_label.shape}\n"
          f"Test_data shape :{test_data.shape}\n"
          f"Test_label shape :{test_label.shape}")

    model = algorithms.GRNN(std=0.4, verbose=True)


    new_test_label = []
    all_length = len(new_df)
    n = split_no

    for i in range(0, all_length - split_no):
        print()
        print(f"No. {i} Model training!  Total number of Training times {all_length - split_no}!\n")
        print(fd_2 + '\n')
        # training model
        a = train_data
        b = train_label
        a = a.reshape((a.shape[0], a.shape[1]))
        b = b.reshape(b.shape[0], 1)
        print(a.shape, b.shape , '\n')

        model.train(a, b)

        # predicting
        x = test_data[i, :]
        x = x.reshape(1, x.shape[0])
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
                    rmse, mape = get_result(full_path_2, fd, fd_2)
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
path = 'D:/Time_Series_Research/new_data/ALL_DATA'
output = find_fd(path)

output.to_csv(os.path.join(path, 'GRNN_result_all_feature.csv'), index=0, header=1)

print_time("program completed in", stime)