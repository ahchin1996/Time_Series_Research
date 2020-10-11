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
