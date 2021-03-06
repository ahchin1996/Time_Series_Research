# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:59:21 2019

@author: f3k11
"""
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
import numpy as np

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# load dataset
dataset = read_csv('D:/Data/Stock/denque1.csv', header=0 , index_col=0)
values = dataset.values
dataset.info()
print(dataset.isnull().sum())
# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# specify the number of lag hours
n_hours = 1
n_features = 9
# frame as supervised learning
reframed = series_to_supervised(scaled, n_hours, 1)
print(reframed.shape)

# split into train and test sets
values = reframed.values
#n_train_hours = 256
#train = values[:n_train_hours, :]
#test = values[n_train_hours:, :]
rumtime=10
score=[[0]*4 for i in range(rumtime)]

for i in range(rumtime): 
    train, test = train_test_split(values, random_state=0, train_size=0.985)
    print(train)
    print(test)
    # split into input and outputs
    n_obs = n_hours * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    test_X, test_y = test[:, :n_obs], test[:, -n_features]
    print(train_X.shape, len(train_X), train_y.shape)
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    # plot history
    #pyplot.plot(history.history['loss'], label='train')
    #pyplot.plot(history.history['val_loss'], label='test')
    #pyplot.legend()
    #pyplot.show()
    
    # make a prediction
    #yhat = model.predict(test_X)
    #test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
    #print(test_X,test_y)
    # invert scaling for forecast
    #inv_yhat = concatenate((yhat, test_X[:, -18:]), axis=1)
    #inv_yhat = scaler.inverse_transform(inv_yhat)
    #inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    #test_y = test_y.reshape((len(test_y), 1))
    #inv_y = concatenate((test_y, test_X[:, -18:]), axis=1)
    #inv_y = scaler.inverse_transform(inv_y)
    #inv_y = inv_y[:,0]
    # 回復Training資料值為原始數據的規模
    trainPredict = model.predict(train_X)
    train_X = train_X.reshape((train_X.shape[0], n_hours*n_features))
    # invert scaling for forecast
    trainPredict = concatenate((trainPredict, train_X[:,  -(n_features-1):]), axis=1)
    trainPredict = scaler.inverse_transform(trainPredict)
    trainPredict = trainPredict[:,0]
    # invert scaling for actual
    train_y = train_y.reshape((len(train_y), 1))
    train_y = concatenate((train_y, train_X[:,  -(n_features-1):]), axis=1)
    train_y = scaler.inverse_transform(train_y)
    train_y = train_y[:,0]
    
    
    # 回復Testing資料值為原始數據的規模
    testPredict = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
    # invert scaling for forecast
    testPredict = concatenate((testPredict, test_X[:, -(n_features-1):]), axis=1)
    testPredict = scaler.inverse_transform(testPredict)
    testPredict = testPredict[:,0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    test_y = concatenate((test_y, test_X[:,  -(n_features-1):]), axis=1)
    test_y = scaler.inverse_transform(test_y)
    test_y = test_y[:,0]

    # calculate RMSE
    
    #calculate 均方根誤差(root mean squared error)
    trainScore = sqrt(mean_squared_error(train_y, trainPredict))
    print('Train RMSE: %.4f' % (trainScore))
    
    testScore = sqrt(mean_squared_error(test_y, testPredict))
    print('Test RMSE: %.4f' % (testScore))
    n = len(test_y)
    mape = sum(np.abs((test_y - testPredict)/test_y))/n*100
    print('Test MAPE: %.4f' % (mape))
    
    pyplot.plot(test_y, color =  'red', label = 'Real Denque Case')  # 紅線表示真實登革熱數
    pyplot.plot(testPredict, color = 'blue', label = 'Predicted Denque Case')  # 藍線表示預測登革熱數
    pyplot.xlabel('Time')
    pyplot.ylabel('dengue case')
    pyplot.legend()
    pyplot.show()

    
    score[i][0]=trainScore
    score[i][1]=testScore
    score[i][2]=trainPredict.tolist()
    score[i][3]=testPredict.tolist()



#rmse = sqrt(mean_squared_error(test_y, testPredict))
#print('Test RMSE: %.3f' % rmse)
