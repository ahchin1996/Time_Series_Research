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
from keras.models import Sequential ,Model,load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from keras import layers

fd_path = 'D:/Time_Series_Research/new_data/DJI/DJI_2000.csv'
df = pd.read_csv(fd_path,sep=',',header=0)
date_array = pd.to_datetime(df['Date'] )
print("Number of rows and columns:", df.shape)

id = 0
while date_array.iloc[id] < datetime(2000, 11, 1, 0, 0):
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
training_data = training_data.reshape((training_data.shape[0],training_data.shape[1]))
test_data = test_data.reshape((test_data.shape[0],test_data.shape[1]))



print(f"Train_data shape : {training_data.shape}\n"
      f"Train_label shape :{training_label.shape}\n"
      f"Test_data shape :{test_data.shape}\n"
      f"Test_label shape :{test_label.shape}")

# 控制顯卡內核
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True
sess0 = tf.compat.v1.InteractiveSession(config=config)

# inputs = (training_data.shape[1],training_data.shape[2])
# model = load_model('D:/Data/Model_LSTM.h5')

custom_early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=200,
    min_delta=0.001,
    mode='auto'
)

modelANN = Sequential()
modelANN.add(layers.Dense(64, input_shape=(training_data.shape[1],), activation="sigmoid"))
modelANN.add(layers.Dropout(0.30))
modelANN.add(layers.Dense(32, activation="sigmoid"))
modelANN.add(layers.Dropout(0.30))
modelANN.add(layers.Dense(16, activation="sigmoid"))
modelANN.add(layers.Dropout(0.30))
modelANN.add(layers.Dense(8, activation="sigmoid"))
modelANN.add(layers.Dropout(0.30))
modelANN.add(layers.Dense(1, activation="sigmoid"))

modelANN.compile(optimizer ='adam', loss ='mean_squared_error', metrics=['accuracy'])
# model.save('Model_LSTM.h5')

modelANN.summary()
# fit network
history = modelANN.fit(training_data,
                       training_label,
                       epochs=1000,
                       batch_size=128,
                       validation_data=(test_data, test_label), verbose=2, shuffle=False,
                       callbacks=[custom_early_stopping])

trainPredict = modelANN.predict(training_data)
testPredict = modelANN.predict(test_data)

# trainScore = sqrt(mean_squared_error(training_label, trainPredict))
# print('Train RMSE: %.4f' % (trainScore))

testPredict = sc_test_label.inverse_transform(testPredict)
test_data = test_data.reshape((test_data.shape[0],test_data.shape[1],))
test_data = sc_test_data.inverse_transform(test_data)
test_label = sc_test_label.inverse_transform(test_label)

testScore = sqrt(mean_squared_error(test_label, testPredict))
print('Test RMSE: %.4f' % (testScore))


n = len(test_label)
testPredict = testPredict.reshape((n,1))
mape = sum(np.abs((test_label - testPredict)/test_label))/n*100
print('Test MAPE: %.4f' % (mape))

flg ,ax = plt.subplots(1,1)
plt.plot(date_array[id:],test_label, color = 'red', label = 'Real Stock Price')
plt.plot(date_array[id:],testPredict, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.xticks(rotation = 30)
plt.ylabel('Stock Price')
plt.legend()
plt.show()

flg ,ax = plt.subplots(1,1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.xticks(rotation = 30)
plt.xlabel('epochs')
plt.legend()
plt.show()
