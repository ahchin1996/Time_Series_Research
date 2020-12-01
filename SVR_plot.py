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

fd_path =  'D:/Time_Series_Research/new_data/HSI/HSI_2001.csv'
df = pd.read_csv(fd_path, sep=',', header=0)
fd_2 = "HSI_2001"

str_in = input("Which feature do you want to choose？")
year = eval(input("Input your year?"))

if str_in.lower() == "all":
    new_df = df
else:
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

#Build the SVR model
regressor = SVR(kernel = 'rbf',gamma='auto',epsilon = 0.01,C= 6,verbose=3)

new_test_label = []
all_length = len(new_df)

for i in range(0, all_length - split_no):
    print()
    print(f"No. {i + 1} Model training! Total number of times {all_length - split_no}!\n")
    print(fd_path + "\n")
    #training model
    a = train_data
    b = train_label
    print(a.shape, b.shape)
    regressor.fit(a,b)

    # fit network
    # predicting
    x = test_data[i, :]
    x = x.reshape(1,x.shape[0])
    testPredict = regressor.predict(x)
    # new_test_label = np.concatenate([new_test_label,testPredict],axis = 0)
    new_test_label.append(testPredict)

    #add next data
    y =  test_label[i]
    # one_y = one_y.reshape(1, one_y.shape[0])
    train_data = np.concatenate([train_data, x], axis=0)
    train_label = np.append(train_label,y)

new_test_label = np.array(new_test_label)
new_test_set = np.concatenate([test_data, new_test_label], axis=1)
new_test_set = sc_df.inverse_transform(new_test_set)
new_test_label = new_test_set[:, new_test_set.shape[1] - 1]

test_set = sc_df.inverse_transform(test_set)
test_label = test_set[:, 0]

rmse = sqrt(mean_squared_error(test_label, new_test_label))
print('Test RMSE: %.4f' % (rmse))

mape = np.mean(np.abs((test_label - new_test_label) / test_label)) * 100
print('Test MAPE: %.4f' % (mape))

flg ,ax = plt.subplots(1,1)
plt.plot(date_array[split_no:], test_label, color ='red', label ='Real Stock Price')
plt.plot(date_array[split_no:], new_test_label, color ='blue', label ='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.xticks(rotation = 30)
plt.ylabel('Stock Price')
plt.legend()
plt.show()