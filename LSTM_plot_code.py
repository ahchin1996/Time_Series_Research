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
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import LSTM,Dropout,Dense
import tensorflow as tf

# 控制顯卡內核
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True
sess0 = tf.compat.v1.InteractiveSession(config=config)

fd_path = 'D:/Time_Series_Research/new_data/GSPC/GSPC_2012.csv'
df = pd.read_csv(fd_path,sep=',',header=0)
date_array = pd.to_datetime(df['Date'] )
print("Number of rows and columns:", df.shape)

id = 0
while date_array.iloc[id] < datetime(2000, 11, 1, 0, 0):
    id +=1
print(id)

df.drop(['Date'] ,axis=1 ,inplace=True)
df.head(5)


sc_df = MinMaxScaler(feature_range = (0, 1))
df = sc_df.fit_transform(df)

train_set = df[:id,:]
test_set = df[id:, :]

train_data = train_set[:,1:]
train_label = train_set[:,0]
test_data = test_set[:,1:]
test_label = test_set[:,0]

train_data = train_data.reshape((train_data.shape[0], train_data.shape[1], 1))
train_label = train_label.reshape(train_label.shape[0], 1)
test_data = test_data.reshape((test_data.shape[0], test_data.shape[1], 1))

print(f"Train_data shape : {train_data.shape}\n"
      f"Train_label shape :{train_label.shape}\n"
      f"Test_data shape :{test_data.shape}\n"
      f"Test_label shape :{test_label.shape}")

# training_set = df.iloc[:id,:]
# test_set = df.iloc[id:,:]
# training_set, test_set = np.array(training_set), np.array(test_set)
#
# #訓練資列統一做歸一後再拆分
# sc_train = MinMaxScaler(feature_range = (0, 1))
#
# training_set_scaled = sc_train.fit_transform(training_set)
#
# train_data = training_set_scaled[:, 1:]
# train_label = training_set_scaled[:, 0]
#
# #測試資料先拆分在做歸一化
# test_data = test_set[:,1:]
# test_label = test_set[:,0]
#
# #label先做reshape
# train_label = train_label.reshape(train_label.shape[0], 1)
# test_label = test_label.reshape(test_label.shape[0],1)
#
# sc_test_data = MinMaxScaler(feature_range = (0, 1))
# test_data = sc_test_data.fit_transform(test_data)
#
# sc_test_label = MinMaxScaler(feature_range = (0, 1))
# test_label = sc_test_label.fit_transform(test_label)
#
# #reshape
# train_data = train_data.reshape((train_data.shape[0], train_data.shape[1], 1))
# test_data = test_data.reshape((test_data.shape[0],test_data.shape[1],1))

# inputs = (training_data.shape[1],training_data.shape[2])
# model = load_model('D:/Data/Model_LSTM.h5')

custom_early_stopping = EarlyStopping(
    monitor='loss',
    patience=100,
    min_delta=0.001,
    mode='auto'
)

params = {'optimizer': ['Adam', "Adadelta"],
     'losses': ['mean_squared_error'],
     'first_hidden_layer': [10, 8, 6],
     'second_hidden_layer': [2, 4, 6],
     'batch_size': [16, 32, 64],
     'epochs': [50, 100]}

def LSTM_model(train_data,train_label,params):
    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape=(train_data.shape[1], train_data.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))

    model.compile(optimizer = params['optimizer'], loss = params['losses'],metrics=['accuracy'])
    # model.save('Model_LSTM.h5')

    model.summary()
    # fit network
    history = model.fit(train_data,
                        train_label,
                        epochs=params['epochs'],
                        batch_size=params['batch_size'],
                        verbose=2, shuffle=False,
                        callbacks=[custom_early_stopping])

    return  history,model

from talos import Scan
h = Scan(X_train_resampled,
         y_train_resampled,
         model=fraud_model,
         params=p,
         grid_downsample=0.1,
         print_params=True,
         dataset_name="creditcardfraud",
         experiment_no='1',
         reduction_metric="val_loss",
         reduce_loss=True)

from talos import Evaluate

e = ta.Evaluate(h)
evaluation = e.evaluate(X_test,
                        y_test,
                        model_id=None,
                        folds=folds,
                        shuffle=True,
                        metric='val_loss',
                        asc=True)

# trainPredict = model.predict(training_data)
testPredict = model.predict(test_data)

new_test_label = testPredict.reshape(testPredict.shape[0],1)
test_data = test_data.reshape(test_data.shape[0],test_data.shape[1])
new_test = np.concatenate([test_data,new_test_label],axis = 1)
new_test = sc_df.inverse_transform(new_test)
new_test_label = new_test[:,23]
# trainScore = sqrt(mean_squared_error(training_label, trainPredict))
# print('Train RMSE: %.4f' % (trainScore))
test_set = sc_df.inverse_transform(test_set)
test_label = test_set[:,0]

testScore = sqrt(mean_squared_error(test_label, new_test_label))
print('Test RMSE: %.4f' % (testScore))


n = len(test_label)
mape = sum(np.abs((test_label - new_test_label)/test_label))/n*100
print('Test MAPE: %.4f' % mape)


flg ,ax = plt.subplots(1,1)
plt.plot(date_array[id:],test_label, color = 'red', label = 'Real Stock Price')
plt.plot(date_array[id:],new_test_label, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.xticks(rotation = 30)
plt.ylabel('Stock Price')
plt.legend()
flg.show()
flg.savefig('fig_cat.png')
    
# flg ,ax = plt.subplots(1,1)
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.xticks(rotation = 30)
# plt.xlabel('epochs')
# plt.legend()
# plt.show()

