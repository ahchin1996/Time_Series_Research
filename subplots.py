import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
fd_path = 'D:/Time_Series_Research/new_data_2/DJI/DJI_2000.csv'
df = pd.read_csv(fd_path, sep=',', header=0)
date_array = pd.to_datetime(df['Date'])

new_df = df[['Date', 'Close']]
new_df['diff_1'] = new_df.Close.diff(1)
new_df['diff_2'] = new_df.diff_1.diff(1)
# new_df['diff_2_2'] = new_df['Close'].diff(2)
df_close = new_df['Close']

flg ,ax= plt.subplots(3,1,sharex=False,sharey=False)
flg.set_size_inches(10,10)
flg.subplots_adjust(hspace=0.6, wspace=0.4)
ax[0].plot(date_array,df_close,'b-')
ax[0].grid(True)
ax[0].tick_params(axis='x',rotation = 30)
ax[0].set_title("Pri    ce")
ax[1].plot(date_array, new_df.diff_1, 'r-')
ax[2].plot(date_array, new_df.diff_2, 'g-')
plt.xlabel('Dates')
plt.show()

flg ,(ax0,ax1,ax2) = plt.subplots(3,1,sharex=False,sharey=False)
flg.subplots_adjust(hspace=0.6, wspace=0.4)
ax0.plot(date_array,df_close,'b-')
ax0.grid(True)
ax0.tick_params(axis='x',rotation = 30)
ax1.plot(date_array, new_df.diff_1, 'r-')
ax2.plot(date_array, new_df.diff_2, 'g-')
plt.xlabel('Dates')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

forecast_path = 'D:/Time_Series_Research/remuneration/DJI_2019_LSTM_selected.csv'
origin_path = 'D:/Time_Series_Research/remuneration/origin_data/DJI_2019.csv'
plot_path = 'D:/Time_Series_Research/plot'

dirname, basename = os.path.split(origin_path)
root, extension = os.path.splitext(origin_path)
stockName = dirname.split("/")[-1]
year = int(basename.strip(extension)[-4:])
fd = basename.strip(extension)

origin_data = pd.read_csv(origin_path,sep=',',header=0)
forecast_close = pd.read_csv(forecast_path, sep=",", names=["Close"])

date_array = pd.to_datetime(origin_data['Date'])
origin_data.set_index(origin_data.Date, inplace=True)

split_no = 0
while date_array.iloc[split_no] < datetime(year, 11, 1, 0, 0):
    split_no += 1
print(split_no)

origin_close = origin_data['Close'][split_no:]
forecast_close.set_index(date_array[split_no:],inplace=True)
forecast_close = forecast_close.squeeze()

date_list = forecast_close.index

flg ,ax = plt.subplots(1,1)
plt.plot(date_list,origin_close, color ='red', label ='Real Stock Price')
plt.plot(date_list,forecast_close, color ='blue', label ='Predicted Stock Price')
plt.title(fd + "_LSTM")
plt.xlabel('Time')
plt.xticks(rotation = 30)
plt.ylabel('Stock Price')
plt.legend()
plt.show()


