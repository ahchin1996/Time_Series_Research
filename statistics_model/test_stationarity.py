import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

fd_path = 'D:/Time_Series_Research/new_data/DJI/DJI_2001.csv'
df = pd.read_csv(fd_path,sep=',',header=0)
date_array = pd.to_datetime(df['Date'] )

#plot close price
flg ,ax = plt.subplots(1,1)
# plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Close Prices')
plt.plot(date_array,df['Close'])
plt.xticks(rotation = 30)
plt.title('Closing price')
plt.show()

flg ,ax = plt.subplots(1,1)
df_close = df['Close']
plt.plot(date_array,df['Close'],'k.')
plt.title('Scatter plot of closing price')
plt.show()


# Test for staionarity
def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    # Plot rolling statistics:
    plt.plot(date_array,timeseries, color='blue', label='Original')
    plt.plot(date_array,rolmean, color='red', label='Rolling Mean')
    plt.plot(date_array,rolstd, color='black', label='Rolling Std')
    plt.xticks(rotation=30)
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)

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

