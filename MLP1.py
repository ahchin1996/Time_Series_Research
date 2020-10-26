import pandas as pd
import numpy as np
from keras import layers
from keras import models
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping

Data = pd.read_csv('D:/Time_Series_Research/1.csv',sep=',',header=0)
Data.info()
Data.replace({'North':1,"South":2,"East":3},inplace=True)

X = Data.drop('theunitpriceNTDsquaremeter',axis =1).values
y = Data['theunitpriceNTDsquaremeter'].values

#splitting Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

#standardization scaler - fit&transform on train, fit only on test
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float))
X_test = s_scaler.transform(X_test.astype(np.float))

early_stopping = EarlyStopping(monitor='val_loss', patience=200,restore_best_weights=True)
model = models.Sequential()
model.add(layers.Dense(19,input_shape=(X_train.shape[1],),  activation="sigmoid"))
# model.add(layers.Dropout(0.30))
model.add(layers.Dense(19, activation="sigmoid"))
# model.add(layers.Dropout(0.30))
model.add(layers.Dense(19, activation="sigmoid"))
# model.add(layers.Dropout(0.30))
model.add(layers.Dense(19, activation="sigmoid"))
# model.add(layers.Dropout(0.30))
model.add(layers.Dense(1, activation="sigmoid"))

model.summary()
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mae'])
model.fit(X_train, y_train, epochs=100, batch_size=128, verbose=2, shuffle=False, callbacks=[early_stopping])
testPredict = model.predict(X_test)

y_test = y_test.reshape(y_test.shape[0],1)

rmse = sqrt(mean_squared_error(y_test, testPredict))
print('Test RMSE: %.4f' % (rmse))

mape = sum(np.abs((y_test - testPredict) / y_test)) / 164 * 100
print('Test MAPE: %.4f' % (mape))

