from datetime import datetime
import os
from math import sqrt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from feature_list import chose_list

# control the kernal inside of GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True
sess0 = tf.compat.v1.InteractiveSession(config=config)

fd_path =  'D:/Time_Series_Research/new_data/GSPC/GSPC_2001.csv'
df = pd.read_csv(fd_path, sep=',', header=0)
fd_2 = "GSPC_2001"

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

#Build the SVR model
regressor = SVR(kernel = 'rbf',gamma='auto',epsilon = 0.01,C= 6,verbose=3)

new_test_label = []
all_length = len(new_df)
n = split_no

for i in range(0, all_length - split_no):
    print()
    print(f"No. {i + 1} Model training! Total number of times {all_length - split_no}!\n")
    print(fd_path + "\n")
    #training model
    a = x_train
    # a = a.reshape((a.shape[0], a.shape[1]))
    b = y_train
    # b = b.reshape((b.shape[0], 1))
    print(a.shape, b.shape)
    regressor.fit(a,b)

    # fit network
    # predicting
    one_x = x_test[i, :]
    one_x = one_x.reshape(1,one_x.shape[0])
    testPredict = regressor.predict(one_x)
    # new_test_label = np.concatenate([new_test_label,testPredict],axis = 0)
    new_test_label.append(testPredict)

    #add next data
    one_y = y_test[i]
    # one_y = one_y.reshape(1, one_y.shape[0])
    x_train = np.concatenate([x_train, one_x], axis=0)
    y_train = np.append(y_train,one_y)

y_test = y_test.reshape(y_test.shape[0],1)
y_test = sc_y.inverse_transform(y_test)
new_test_label = sc_y.inverse_transform(new_test_label)

testScore = sqrt(mean_squared_error(y_test, new_test_label))
print('Test RMSE: %.4f' % (testScore))

mape = np.mean(np.abs((y_test - new_test_label)/y_test) )*100
print('Test MAPE: %.4f' % (mape))

flg ,ax = plt.subplots(1,1)
plt.plot(date_array[split_no:], y_test, color ='red', label ='Real Stock Price')
plt.plot(date_array[split_no:], new_test_label, color ='blue', label ='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.xticks(rotation = 30)
plt.ylabel('Stock Price')
plt.legend()
plt.show()