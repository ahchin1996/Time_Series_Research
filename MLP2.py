import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

Data = pd.read_csv('D:/Time_Series_Research/1.csv',sep=',',header=0)
Data.info()
Data.replace({'North':1,"South":2,"East":3},inplace=True)

X = Data.drop('theunitpriceNTDsquaremeter',axis =1).values
y = Data['theunitpriceNTDsquaremeter'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

#standardization scaler - fit&transform on train, fit only on test
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float))
X_test = s_scaler.transform(X_test.astype(np.float))

# Multiple Liner Regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set result
y_pred = regressor.predict(X_test)


rmse = sqrt(mean_squared_error(y_test, y_pred))
print('Test RMSE: %.4f' % (rmse))

mape = np.mean(np.abs((y_test - y_pred)/y_test) )*100
print('Test MAPE: %.4f' % (mape))

