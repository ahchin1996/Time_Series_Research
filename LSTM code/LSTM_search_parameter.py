# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 22:19:16 2020

@author: cooke
"""

from datetime import datetime
import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, Flatten
from feature_list import *
import talos as ta
import time
import tensorflow as tf

# hide INFO and WARNING message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# assgin which one GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# control the kernal inside of GPU
config = tf.compat.v1.ConfigProto()
# 指定GPU顯示卡記憶體用量上限
config.gpu_options.per_process_gpu_memory_fraction = .75
# automatic selection running device
config.allow_soft_placement = True
# 自動增長GPU記憶體用量
config.gpu_options.allow_growth = True
sess = tf.compat.v1.InteractiveSession(config=config)

stime = time.time()

def print_time(text, stime):
    seconds = (time.time() - stime)
    print()
    print(text +" "+ str(seconds // 60 // 60)+" hours : " + str(seconds // 60 % 60)  + " minutes : " + str(np.round(seconds % 60)) + " seconds")

#每次需更改項目
year = 2011
fd = 'TWII_2011'
path =  'D:/Time_Series_Research/new_data/TWII/TWII_2011.csv'

df_all = pd.read_csv(path,sep=',',header=0)
date_array = pd.to_datetime(df_all['Date'] )
print("Number of rows and columns:", df_all.shape)

feature_list = chose_list(fd)
new_df = df_all[['Date', 'Close']]
for i in range(0, len(feature_list)):
    new_df = pd.concat([new_df, df_all.iloc[:, feature_list[i]]], axis=1)

new_df.columns
print("Number of rows and columns:", new_df.shape)

split_no = 0
while date_array.iloc[split_no] < datetime(year, 11, 1, 0, 0):
    split_no +=1
print(split_no)

new_df.drop(['Date'], axis=1, inplace=True)
new_df.head(5)

new_df_orign = pd.DataFrame(new_df)

sc_df = MinMaxScaler(feature_range = (0, 1))
new_df = sc_df.fit_transform(new_df)

train_set = new_df[:split_no, :]
test_set = new_df[split_no:, :]

train_data = train_set[:,1:]
train_label = train_set[:,0]
test_data = test_set[:,1:]
test_label = test_set[:,0]

print(f"Train_data shape : {train_data.shape}\n"
f"Train_label shape :{train_label.shape}\n"
f"Test_data shape :{test_data.shape}\n"
f"Test_label shape :{test_label.shape}")

x_train = train_data[:-5,:]
x_val = train_data[-5:,:]

y_train = train_label[:-5]
y_val = train_label[-5:]

######################################
x_train = x_train.reshape(x_train.shape[0],1,x_train.shape[1])
x_val = np.reshape(x_val,(x_val.shape[0],1,x_val.shape[1]))

y_train = np.reshape(y_train,(y_train.shape[0],1))
y_val = np.reshape(y_val,(y_val.shape[0],1))

print(f"x_train shape : {x_train.shape}\n"
f"x_val shape :{x_val.shape}\n"
f"y_train shape :{y_train.shape}\n"
f"y_val shape :{y_val.shape}")

search_params = {
"lstm_layers": [1,2],
"lstm_1_nodes" : [30, 50,70],
"lstm_2_nodes" : [30, 50,70],
"batch_size": [64,128],
"lr": [0.1, 0.01, 0.001],
"epochs": [50,80],
"optimizer": ['Adam']
}

def create_model_talos(train_data, train_label, x_val, y_val, params):
    BATCH_SIZE = params["batch_size"]
    EPOCHS = params["epochs"]
    lstm_model = Sequential()
    # (batch_size, timesteps, data_dim)
    lstm_model.add(LSTM(params["lstm_1_nodes"], input_shape=(1 ,train_data.shape[2]), return_sequences=True))

    if params["lstm_layers"] == 2:
        lstm_model.add(LSTM(params["lstm_2_nodes"], return_sequences = True))
        lstm_model.add(Flatten())
    else:
        lstm_model.add(Flatten())

    lstm_model.add(Dense(1))

    if params["optimizer"] == 'Adam':
        optimizer = Adam(lr=params["lr"])

    lstm_model.compile(loss='mean_squared_error', optimizer=optimizer, metrics = ['acc'])  # binary_crossentropy
    history = lstm_model.fit(train_data,
                             train_label,
                             epochs = EPOCHS,
                             verbose=2,
                             batch_size=BATCH_SIZE,
                             validation_data=[x_val,y_val]
                             )
    # for key in history.history.keys():
    #     print(key, "--",history.history[key])
    print_time("program running in", stime)
    print()
    return history, lstm_model


print("Starting Talos scanning...")
t = ta.Scan(x= x_train,
            y= y_train,
            x_val= x_val,
            y_val= y_val,
            model=create_model_talos,
            params=search_params,
            experiment_name = "LSTM_parameter_result",
            print_params = True,
            clear_session= True)

print()
print_time("program completed in", stime)

from numba import cuda
device = cuda.get_current_device()
device.reset()

import gc
gc.collect()