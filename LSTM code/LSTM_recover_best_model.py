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
from feature_list import chose_list
import talos as ta
import time
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.callbacks import EarlyStopping

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
    print(text +" "+ str(seconds // 60 // 60)+" hour(s) : " + str(seconds // 60 % 60)  + " minute(s) : " + str(np.round(seconds % 60)) + " second(s)")

#每次需更改項目
year = 2011
fd = 'TWII_2011'
path =  'D:/Time_Series_Research/new_data/TWII/TWII_2011.csv'
repot_path = 'D:/Time_Series_Research/LSTM code/LSTM_parameter_result/TWII_2011_p.csv'


INPUT_PATH = os.path.join(path, "inputs")

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

x_train = np.reshape(train_data,(train_data.shape[0],1,train_data.shape[1]))
y_train = np.reshape(train_label,(train_label.shape[0],1))


print(f"Train_data shape : {train_data.shape}\n"
f"Train_label shape :{train_label.shape}\n"
f"Test_data shape :{test_data.shape}\n"
f"Test_label shape :{test_label.shape}")

custom_early_stopping = EarlyStopping(
    monitor='loss',
    patience=100,
    min_delta=0.01,
    mode='auto'
)

def create_model_talos(train_data, params):
    lstm_model = Sequential()
    # (batch_size, timesteps, data_dim)
    lstm_model.add(LSTM(params["lstm_1_nodes"], input_shape=(1, train_data.shape[2]), return_sequences=True))

    if params["lstm_layers"] == 2:
        lstm_model.add(LSTM(params["lstm_2_nodes"], return_sequences=True))
        lstm_model.add(Flatten())
    else:
        lstm_model.add(Flatten())

    lstm_model.add(Dense(1))

    if params["optimizer"] == 'Adam':
        optimizer = Adam(lr=params["lr"])

    lstm_model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['acc'])  # binary_crossentropy
    # for key in history.history.keys():
    #     print(key, "--",history.history[key])
    return lstm_model

r = ta.Reporting(repot_path)
p_list = r.data
low_val_loss =  r.low('val_loss')
fliter = p_list.val_loss == low_val_loss
best_p = p_list[fliter]

par_id = best_p.index

print(best_p)
print(par_id)

best_p = best_p.to_dict(orient = "list")

newDict= {k: v[0] for k, v in best_p.items()}

lstm_model = create_model_talos(x_train,newDict)
lstm_model.summary()

a = train_data
b = train_label
a = a.reshape(a.shape[0],1, a.shape[1])
b = b.reshape(b.shape[0], )

new_test_label = []
all_length = len(new_df)
n = split_no

for i in range(0, all_length - split_no):
    print()
    print(f"No. {i + 1} Model training! Total number of times {all_length - split_no}!\n")
    print(path + "\n")
    #training model

    print(a.shape, b.shape)
    lstm_model.fit(a,
                   b,
                   epochs=newDict["epochs"],
                   batch_size=newDict["batch_size"],
                   verbose=2,
                   shuffle=False,
                   callbacks= [custom_early_stopping]
                  )

    # fit network
    #predicting
    x = test_data[i, :]
    x = x.reshape(1,1,x.shape[0])
    testPredict = lstm_model.predict(x)
    # new_test_label = np.concatenate([new_test_label,testPredict],axis = 0)
    new_test_label.append(testPredict)

    #add next data
    y = test_data[i, :]
    y = y.reshape(1, 1, y.shape[0])
    train_data = np.concatenate([a, y], axis=0)
    z = test_label[i].reshape(1, )
    train_label = np.concatenate([b, z], axis=0)

new_test_label = np.array(new_test_label)
new_test_label = new_test_label.reshape(new_test_label.shape[0], 1)
new_test_set = np.concatenate([test_data, new_test_label], axis=1)
new_test_set = sc_df.inverse_transform(new_test_set)
new_test_label = new_test_set[:, new_test_set.shape[1] - 1]

test_label = np.array(new_df_orign.iloc[split_no:,0])

testScore = sqrt(mean_squared_error(test_label, new_test_label))
print('Test RMSE: %.4f' % (testScore))

mape = np.mean(np.abs((test_label - new_test_label)/test_label) )*100
print('Test MAPE: %.4f' % (mape))

print_time("program completed in", stime)

print(fd)
print(best_p)
print(par_id)