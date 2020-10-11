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
from neupy import algorithms
from statsmodels.tsa.arima_model import ARIMA

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
while date_array.iloc[id] < datetime(2012, 11, 1, 0, 0):
    id +=1
print(id)

df.drop(['Date'] ,axis=1 ,inplace=True)
df.head(5)
df.describe()

sc_df = MinMaxScaler(feature_range = (0, 1))
df = sc_df.fit_transform(df)

train_set = df[:id,:]
test_set = df[id:, :]

train_data = train_set[:,1:]
train_label = train_set[:,0]
test_data = test_set[:,1:]
test_label = test_set[:,0]

print(f"Train_data shape : {train_data.shape}\n"
      f"Train_label shape :{train_label.shape}\n"
      f"Test_data shape :{test_data.shape}\n"
      f"Test_label shape :{test_label.shape}")

custom_early_stopping = EarlyStopping(
    monitor='loss',
    patience=100,
    min_delta=0.001,
    mode='auto'
)

# fit model
model = ARIMA(data, order=(1, 1, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(len(data), len(data), typ='levels')
print(yhat)


# model.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=['accuracy'])
# # model.save('Model_LSTM.h5')
#
# model.summary()
#     # fit network
# history = model.fit(train_data,
#                     train_label,
#                     epochs=100,
#                     batch_size=64,
#                     verbose=2, shuffle=False,
#                     callbacks=[custom_early_stopping])
#
# # trainPredict = model.predict(training_data)
# new_test_label = model.predict(test_data)

new_test_label = np.array(new_test_label)
new_test_label = new_test_label.reshape(new_test_label.shape[0],1)
new_test_set = np.concatenate([test_data,new_test_label],axis = 1)
new_test_set = sc_df.inverse_transform(new_test_set)
new_test_label = new_test_set[:,new_test_set.shape[1]-1]

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
# flg.savefig('fig_cat.png')
    
# flg ,ax = plt.subplots(1,1)
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.xticks(rotation = 30)
# plt.xlabel('epochs')
# plt.legend()
# plt.show()

