import os
from datetime import datetime
from math import sqrt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from sklearn.svm import SVR
from feature_list import chose_list
import matplotlib.pyplot as plt

# hide INFO and WARNING message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# control the kernal inside of GPU
# assgin which one GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# automatic selection running device
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
# 指定GPU顯示卡記憶體用量上限
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
# 自動增長GPU記憶體用量
config.gpu_options.allow_growth = True
sess0 = tf.compat.v1.InteractiveSession(config=config)

#每次需更改項目
year = 2019
fd = 'DJI_2019'
path =  'D:/Time_Series_Research/new_data/DJI/DJI_2019.csv'

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
regressor = SVR(kernel = 'linear',gamma='scale',epsilon = 0.0001,C= 8,verbose=3)

new_test_label = []
all_length = len(new_df)
for i in range(0, all_length - split_no):
    print()
    print(f"No. {i + 1} Model training! Total number of times {all_length - split_no}!\n")
    print(path + "\n")
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

test_set = np.array(new_df_orign.iloc[split_no:, :])
test_label = test_set[:, 0]

rmse = sqrt(mean_squared_error(test_label, new_test_label))
print('Test RMSE: %.4f' % (rmse))

mape = np.mean(np.abs((test_label - new_test_label) / test_label)) * 100
print('Test MAPE: %.4f' % (mape))

# flg ,ax = plt.subplots(1,1)
# plt.plot(date_array[split_no:], test_label, color ='red', label ='Real Stock Price')
# plt.plot(date_array[split_no:], new_test_label, color ='blue', label ='Predicted Stock Price')
# plt.title('Stock Price Prediction')
# plt.xlabel('Time')
# plt.xticks(rotation = 30)
# plt.ylabel('Stock Price')
# plt.legend()
# plt.show()