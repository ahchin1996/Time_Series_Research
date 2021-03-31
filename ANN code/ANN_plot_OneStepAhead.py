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
import tensorflow as tf
from keras import layers


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

fd_path =  'D:/Time_Series_Research/new_data/TWII/TWII_2019.csv'

df = pd.read_csv(fd_path, sep=',', header=0)

str_in = input("Which feature do you want to choose？")
year = eval(input("Input your year?"))

if str_in.lower() == "all":
    new_df = df
else:
    choose_feature = [int(n) for n in str_in.split()]
    new_df = df[['Date', 'Close']]
    for i in range(0,len(choose_feature)):
        new_df = pd.concat([new_df, df.iloc[:, choose_feature[i]]], axis=1)

new_df.columns


date_array = pd.to_datetime(new_df['Date'])
print("Number of rows and columns:", new_df.shape)

split_no = 0
while date_array.iloc[split_no] < datetime(year, 11, 1, 0, 0):
    split_no +=1
print(split_no)

new_df.drop(['Date'], axis=1, inplace=True)
new_df.head(5)

sc_df = MinMaxScaler(feature_range = (0, 1))
new_df = sc_df.fit_transform(new_df)

train_set = new_df[:split_no, :]
test_set = new_df[split_no:, :]

train_data = train_set[:,1:]
train_label = train_set[:,0]
test_data = test_set[:,1:]
test_label = test_set[:,0]

d1 = train_label.shape[0]
d2 = test_label.shape[0]
train_label = train_label.reshape((d1,1))
test_label = test_label.reshape((d2,1))

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
#Built Model
model = Sequential()
model.add(layers.Dense(64, input_shape=(train_data.shape[1],), activation="sigmoid"))
model.add(layers.Dropout(0.30))
model.add(layers.Dense(32, activation="sigmoid"))
model.add(layers.Dropout(0.30))
model.add(layers.Dense(16, activation="sigmoid"))
model.add(layers.Dropout(0.30))
model.add(layers.Dense(8, activation="sigmoid"))
model.add(layers.Dropout(0.30))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=['accuracy'])
# model.save('Model_LSTM.h5')

model.summary()

new_test_label = []
all_length = len(new_df)
n = split_no

for i in range(0, all_length - split_no):
    print()
    print(f"No. {i + 1} Model training! Total number of times {all_length - split_no}!")
    print()
    #training model
    a = train_data
    b = train_label
    a = a.reshape((a.shape[0], a.shape[1]))
    b = b.reshape(b.shape[0], 1)
    print(a.shape, b.shape)
    model.fit(a,
              b,
              epochs=100,
              batch_size=64,
              verbose=2, shuffle=False,
              callbacks=[custom_early_stopping])

    # fit network
    #predicting
    x = test_data[i, :]
    x = x.reshape(1,x.shape[0])
    testPredict = model.predict(x)
    # new_test_label = np.concatenate([new_test_label,testPredict],axis = 0)
    new_test_label.append(testPredict)

    #add next data
    y = test_data[i, :]
    y = y.reshape(1,y.shape[0])
    train_data = np.concatenate([train_data,y],axis = 0)
    z = test_label[i].reshape(1,1)
    train_label = np.concatenate([train_label, z], axis=0)

new_test_label = np.array(new_test_label)
new_test_label = new_test_label.reshape(new_test_label.shape[0],1)
new_test_set = np.concatenate([test_data,new_test_label],axis = 1)
new_test_set = sc_df.inverse_transform(new_test_set)
new_test_label = new_test_set[:,new_test_set.shape[1]-1]

test_set = sc_df.inverse_transform(test_set)
test_label = test_set[:,0]

testScore = sqrt(mean_squared_error(test_label, new_test_label))
print('Test RMSE: %.4f' % (testScore))

mape = sum(np.abs((test_label - new_test_label)/test_label))/n*100
print('Test MAPE: %.4f' % (mape))

flg ,ax = plt.subplots(1,1 ,)
plt.plot(date_array[split_no:], test_label, color ='red', label ='Real Stock Price')
plt.plot(date_array[split_no:], new_test_label, color ='blue', label ='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.xticks(rotation = 30)
plt.ylabel('Stock Price')
plt.legend()
# plt.show()
plt.savefig('D:\Time_Series_Research\plot\\example.png',bbox_inches = 'tight')
plt.close()

# # flg ,ax = plt.subplots(1,1)
# # plt.plot(history.history['loss'], label='train')
# # plt.plot(history.history['val_loss'], label='test')
# # plt.xticks(rotation = 30)
# # plt.xlabel('epochs')
# # plt.legend()
# # plt.show()

