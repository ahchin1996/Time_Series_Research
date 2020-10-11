#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ARIMA Model
from statsmodels.tsa.arima_model import ARIMA
# from statsmodels.tsa.statespace.varmax import VARMAX
import statsmodels.api as sm
from random import random
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from datetime import datetime
import os
from math import sqrt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
import pyflux as pf


# In[2]:


fd_path = 'D:/Time_Series_Research/new_data/DJI/DJI_2004.csv'
df = pd.read_csv(fd_path, sep=',', header=0)


# In[3]:


df.dtypes


# In[4]:


str_in = input("Whic feature you want to choose？")
year = eval(input("Input your year?"))


# In[5]:


if str_in.strip().lower() == "all":
    new_df = df
else:
    choose_feature = [int(n) for n in str_in.split()]
    new_df = df[['Date', 'Close']]
    for i in range(0,len(choose_feature)):
        new_df = pd.concat([new_df, df.iloc[:, choose_feature[i]]], axis=1)

print(new_df.columns,'\n')

date_array = pd.to_datetime(new_df['Date'])
print("Number of rows and columns:", new_df.shape,'\n')

split_no = 0
while date_array.iloc[split_no] < datetime(year, 11,1, 0, 0):
    split_no +=1
print(split_no,'\n')

new_df.drop(['Date'], axis=1, inplace=True)
new_df.head(5)

# sc_df = MinMaxScaler(feature_range = (0, 1))
# new_df = sc_df.fit_transform(new_df)

new_df = new_df.values
train_set = new_df[:split_no, :]
test_set = new_df[split_no:, :]

train_data = train_set[:,1:]
train_label = train_set[:,0]
test_data = test_set[:,1:]
test_label = test_set[:,0]


# In[6]:


# train_label = train_label.reshape(train_label.shape[0],1)
# test_label = test_label.reshape(test_label.shape[0],1)
print(f"Train_data shape : {train_data.shape}\n"
      f"Train_label shape :{train_label.shape}\n"
      f"Test_data shape :{test_data.shape}\n"
      f"Test_label shape :{test_label.shape}\n")


# In[7]:


#原始函數
df_close = df['Close']
df_close.index = date_array
df_close.head()


# In[8]:


#p 和 q 階數確定，ACF和PACF
flg , ax = plt.subplots(2,1,sharex = False ,sharey = False)
flg.set_size_inches(8, 8)
flg.subplots_adjust(hspace=0.6, wspace=0.4)
flg= plot_acf(df_close, lags= 20 ,ax=ax[0])
flg= plot_pacf(df_close, lags= 12 ,ax=ax[1])
plt.show()


# In[9]:


sm.tsa.stattools.adfuller(df_close)[:2]


# In[10]:


# Test for staionarity
def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    # Plot rolling statistics:
    plt.plot(date_array,timeseries, color='blue', label='Original')
    plt.plot(date_array,rolmean, color='red', label='Rolling Mean')
#     plt.plot(date_array,rolstd, color='black', label='Rolling Std')
    plt.xticks(rotation=30)
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)
    
    # Test for staionarity
    print("Results of dickey fuller test")
    adft = adfuller(timeseries, autolag='AIC')
    # output for dft will give us without defining what the values are.
    # hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],
                       index=['Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])
    for key, values in adft[4].items():
        output['critical value (%s)' % key] = values
    print(output)


test_stationarity(df_close)


# In[11]:


#一階方差、二階方差
diff1 = df_close.diff(1)[1:]
diff2 = diff1.diff(1)[1:]


# In[12]:


#將序列平穩(差分法確定d)
flg ,ax= plt.subplots(3,1,sharex=False,sharey=False)
flg.set_size_inches(10,10)
flg.subplots_adjust(hspace=0.6, wspace=0.4)
ax[0].plot(date_array,df_close,'b-',label = 'price')
ax[0].grid(True)
ax[0].tick_params(axis='x',rotation = 30)
ax[0].set_title("Price")
# ax[0].legend(loc='best')
ax[1].plot(diff1, 'r-')
ax[1].set_title("Diff 1")
ax[2].plot(diff2, 'g-')
ax[2].set_title("Diff 2")
plt.xlabel('Dates')
plt.show()


# In[13]:


print(sm.tsa.stattools.adfuller(diff1)[:2])
print(sm.tsa.stattools.adfuller(diff2)[:2])


# In[14]:


#一階方差，ACF和PACF
flg , ax = plt.subplots(2,1,sharex = False ,sharey = False)
flg.set_size_inches(8, 8)
flg.subplots_adjust(hspace=0.6, wspace=0.4)
flg= plot_acf(diff1, lags= 20 ,ax=ax[0])
flg= plot_pacf(diff1, lags= 12 ,ax=ax[1])
plt.show()


# # ARIMA TEST MODEL 1

# from pmdarima.arima import auto_arima

# model = auto_arima(train_label,exogenous=train_data,start_p=1,d=1,start_q=0)model = auto_arima(train_label,start_p=1,d=1,start_q=0)

# model.fit(test_label)

# model.summary()

# preds, conf_int = model.predict(n_periods=test_label.shape[0], return_conf_int=True)

# new_test_label = np.array(preds)
# new_test_label = new_test_label.reshape(new_test_label.shape[0],1)
# new_test_set = np.concatenate([test_data,new_test_label],axis = 1)
# new_test_set = sc_df.inverse_transform(new_test_set)
# new_test_label = new_test_set[:,new_test_set.shape[1]-1]
# 
# 
# test_label = df.Close[split_no:]
# 
# testScore = sqrt(mean_squared_error(test_label, new_test_label))
# print('Test RMSE: %.4f' % (testScore))
# 
# mape = sum(np.abs((test_label - new_test_label)/test_label))/split_no*100
# print('Test MAPE: %.4f' % (mape))
# 
# flg ,ax = plt.subplots(1,1)
# plt.plot(date_array[split_no:], test_label, color ='red', label ='Real Stock Price')
# plt.plot(date_array[split_no:], new_test_label, color ='blue', label ='Predicted Stock Price')
# plt.title('Stock Price Prediction')
# plt.xlabel('Time')
# plt.xticks(rotation = 30)
# plt.ylabel('Stock Price')
# plt.legend()
# plt.show()
# 

# # ARIMA TEST MODEL 2

arima100 = ARIMA(endog = train_label, exog=train_data  ,order = (0,1,0))

model_fit = arima100.fit(disp = 0)

x = test_data[0, :]
x = x.reshape(1,x.shape[0])
x.shape

forecast = model_fit.forecast(exog= x)
forecast
# In[15]:


new_test_label = []
all_length = len(new_df)
n = split_no


# In[16]:


for i in range(0, all_length - split_no):
    print()
    print(f"No. {i + 1} Model training! Total number of times {all_length - split_no}!\n")
    print(fd_path + "\n")
    #training model
    a = train_data 
    b = train_label

    arima100 = ARIMA(endog = b,exog = a,order = (0,1,0))
    model_fit = arima100.fit(disp = 0)

    # fit network
    #predicting
    x = test_data[i, :]
    x = x.reshape(1,x.shape[0])
    testPredict = model_fit.forecast(exog = x)
    # new_test_label = np.concatenate([new_test_label,testPredict],axis = 0)
    new_test_label.append(testPredict[0])

    #add next data
    y = test_data[i, :]
    y = y.reshape(1, y.shape[0])
    train_data = np.concatenate([train_data, y], axis=0)
    z = test_label[i].reshape(1, )
    train_label = np.concatenate([train_label, z], axis=0)


# In[17]:


new_test_label = np.array(new_test_label)

test_label = test_label.reshape(test_label.shape[0],1)

testScore = sqrt(mean_squared_error(test_label, new_test_label))
print('Test RMSE: %.4f' % (testScore))

mape = sum(np.abs((test_label - new_test_label)/test_label))/split_no*100
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


# In[ ]:




