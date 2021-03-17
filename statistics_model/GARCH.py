import pandas_datareader.data as web
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np


#DIS Volatility
start = datetime(2015, 1, 1)
end = datetime(2020, 6, 10)
dis = web.DataReader('DIS', 'yahoo', start=start, end=end)

returns = 100 * dis.Close.pct_change().dropna()



plt.figure(figsize=(10,4))
plt.plot(returns)
plt.ylabel('Pct Return', fontsize=16)
plt.title('DIS Returns', fontsize=20)

#PACF
plot_pacf(returns**2)
plt.show()

#Fit GARCH(3,3)
model = arch_model(returns, p=3, q=3)
model_fit = model.fit()
model_fit.summary()

#Try GARCH(3,0) = ARCH(3)
model = arch_model(returns, p=3, q=0)
model_fit = model.fit()
model_fit.summary()

rolling_predictions = []
test_size = 365

for i in range(test_size):
    train = returns[:-(test_size-i)]
    model = arch_model(train, p=3, q=0)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))

rolling_predictions = pd.Series(rolling_predictions, index=returns.index[-365:])

plt.figure(figsize=(10,4))
true, = plt.plot(returns[-365:])
preds, = plt.plot(rolling_predictions)
plt.title('Volatility Prediction - Rolling Forecast', fontsize=20)
plt.legend(['True Returns', 'Predicted Volatility'], fontsize=16)


