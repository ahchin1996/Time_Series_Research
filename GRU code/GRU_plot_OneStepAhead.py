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
from keras.layers import  Dense, GRU, Dropout, Flatten
import tensorflow as tf
from feature_list import *
from keras.optimizers import Adam

# hide INFO and WARNING message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# assgin which one GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# control the kernal inside of GPU
config = tf.compat.v1.ConfigProto()
# 指定GPU顯示卡記憶體用量上限
# config.gpu_options.per_process_gpu_memory_fraction = .75
# automatic selection running device
config.allow_soft_placement = True
# 自動增長GPU記憶體用量
config.gpu_options.allow_growth = True
sess = tf.compat.v1.InteractiveSession(config=config)

stime = time.time()

#每次需更改項目
path =  'D:/Time_Series_Research/new_data/ALL_DATA/DJI/DJI_2019.csv'

dirname, basename = os.path.split(path)
root, extension = os.path.splitext(path)
stockName = dirname.split("/")[-1]
year = int(basename.strip(extension)[-4:])
fd = basename.strip(extension)

df_all = pd.read_csv(path,sep=',',header=0)
date_array = pd.to_datetime(df_all['Date'] )
print("Number of rows and columns:", df_all.shape)

# feature_list = chose_list_all()
feature_list = chose_list(fd)

new_df = df_all[['Date', 'Close']]
for i in range(0, len(feature_list)):
    new_df = pd.concat([new_df, df_all.iloc[:, feature_list[i]]], axis=1)

new_df.columns
print("Number of rows and columns:", new_df.shape)

split_no = 0
while date_array.iloc[split_no] < datetime(year, 11,1, 0, 0):
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
      f"Test_label shape :{test_label.shape}\n")

custom_early_stopping = EarlyStopping(
    monitor='loss',
    patience=100,
    min_delta=0.01,
    mode='min'
)

#Built Model
model = Sequential()
model.add(GRU(units=80, return_sequences=True, input_shape=(1,x_train.shape[2])))
# model.add(GRU(units=50, return_sequences=True))
# model.add(GRU(units=30, return_sequences=True))
# model.add(GRU(units=30))
model.add(Flatten())
model.add(Dense(units=1))

# sgd = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
# ad = Adam(learning_rate=0.001)

model.compile(optimizer = Adam(lr = 0.001) , loss = 'mean_squared_error',metrics=['accuracy'])

# model.save('Model_LSTM.h5')

model.summary()

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
    model.fit(a,
              b,
              epochs=80,
              batch_size = 64,
              verbose=2,
              shuffle=False,
              callbacks=[custom_early_stopping])

    # fit network
    #predicting
    x = test_data[i, :]
    x = x.reshape(1,1,x.shape[0])
    testPredict = model.predict(x)
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

print(fd)
testScore = sqrt(mean_squared_error(test_label, new_test_label))
print('Test RMSE: %.4f' % (testScore))

mape = np.mean(np.abs((test_label - new_test_label)/test_label) )*100
print('Test MAPE: %.4f' % (mape))

print_time("program completed in", stime)


new_test_label = pd.DataFrame(new_test_label)
# new_test_label.to_csv("D:/Time_Series_Research/remuneration/"+fd+"_GRU_full.csv", index=0, header=0)
new_test_label.to_csv("D:/Time_Series_Research/remuneration/"+fd+"_GRU_selected.csv", index=0, header=0)

# flg ,ax = plt.subplots(1,1)
# plt.plot(date_array[split_no:], test_label, color ='red', label ='Real Stock Price')
# plt.plot(date_array[split_no:], new_test_label, color ='blue', label ='Predicted Stock Price')
# plt.title(fd+" GRU")
# plt.xlabel('Time')
# plt.xticks(rotation = 30)
# plt.ylabel('Stock Price')
# plt.legend()
# plt.show()
#
# new_test_label = pd.DataFrame(new_test_label)
# new_test_label.to_csv("D:/Time_Series_Research/remuneration/GSPC_2012_GRU_pred.csv", index=0, header=0)
# # flg ,ax = plt.subplots(1,1)
# # plt.plot(history.history['loss'], label='train')
# # plt.plot(history.history['val_loss'], label='test')
# # plt.xticks(rotation = 30)
# # plt.xlabel('epochs')
# # plt.legend()
# # plt.show()

