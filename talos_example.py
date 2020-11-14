import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from feature_list import chose_list
from datetime import datetime
import os
from math import sqrt
import pandas as pd
import numpy as np
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.layers import LSTM,Dropout,Dense
from sklearn.metrics import mean_squared_error

# control the kernal inside of GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True
sess0 = tf.compat.v1.InteractiveSession(config=config)

fd_path =  'D:/Time_Series_Research/new_data/GSPC/GSPC_2019.csv'
df = pd.read_csv(fd_path, sep=',', header=0)
fd_2 = "GSPC_2019"

str_in = input("Which feature do you want to choose？")
year = eval(input("Input your year?"))

if str_in.lower() == "all":
    new_df = df
else:
    # choose_feature = [int(n) for n in str_in.split()]
    # new_df = df[['Date', 'Close']]
    # for i in range(0,len(choose_feature)):
    #     new_df = pd.concat([new_df, df.iloc[:, choose_feature[i]]], axis=1)
    feature_list = chose_list(fd_2)

    new_df = df[['Date', 'Close']]
    for i in range(0, len(feature_list)):
        new_df = pd.concat([new_df, df.iloc[:, feature_list[i]]], axis=1)

new_df.columns

date_array = pd.to_datetime(new_df['Date'])
print("Number of rows and columns:", new_df.shape)

split_no = 0
while date_array.iloc[split_no] < datetime(year, 11, 1, 0, 0):
    split_no +=1
print(split_no)

new_df.drop(['Date'], axis=1, inplace=True)
new_df.head(5)

y = new_df.Close
x = new_df.drop(["Close"],axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - (split_no/len(date_array)),shuffle=False)


sc_x = MinMaxScaler(feature_range = (0, 1))
x_train = sc_x.fit_transform(x_train.astype(np.float))
x_test = sc_x.fit_transform(x_test.astype(np.float))

sc_y = MinMaxScaler(feature_range = (0, 1))
y_train = np.array(y_train)
y_train = y_train.reshape(y_train.shape[0],1)
abc = y_train
y_train = sc_y.fit_transform(y_train.astype(np.float))
scale = sc_y.fit(abc)

y_test = np.array(y_test)
y_test = y_test.reshape(y_test.shape[0],1)
y_test = sc_x.fit_transform(y_test.astype(np.float))

y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])

print(f"Train_data shape : {x_train.shape}\n"
      f"Train_label shape :{y_train.shape}\n"
      f"Test_data shape :{x_test.shape}\n"
      f"Test_label shape :{y_test.shape}")

p = {'activation1':["relu", "elu"],
     'activation2':["relu", "elu"],
     'optimizer': ['Adam', "RMSprop"],
     'losses': ['logcosh', keras.losses.binary_crossentropy],
     'first_hidden_layer': [10, 8, 6],
     'second_hidden_layer': [2, 4, 6],
     'batch_size': [100, 1000, 10000],
     'epochs': [10, 15]}

def fraud_model(X_train, y_train, x_val, y_val, params):
    model = Sequential()
    model.add(Dense(params['first_hidden_layer'],
                    input_shape=(29,),
                    activation=params['activation1'],
                    use_bias=True))
    model.add(Dropout(0.2))
    model.add(Dense(params['second_hidden_layer'],
                    activation=params['activation2'],
                    use_bias=True))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer=params['optimizer'],
                  loss=params['losses'],
                  metrics=[keras.metrics.binary_accuracy])
    history = model.fit(X_train_resampled,
                    y_train_resampled,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    verbose=1,
                    validation_data=[X_val_resampled, y_val_resampled],
                    callbacks=[early_stopper(epochs=params['epochs'],
                                                    mode='moderate',
                                                    monitor='val_loss')])
    return history, model