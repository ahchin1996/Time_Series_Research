import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

forecast_path = 'D:/Time_Series_Research/remuneration/TWII_2019_LSTM_selected.csv'
origin_path = 'D:/Time_Series_Research/remuneration/origin_data/TWII_2019.csv'
path = 'D:/Time_Series_Research/plot/'

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

plot_path = os.path.join(path,fd)

flg ,ax = plt.subplots(1,1)
plt.plot(date_list,origin_close, color ='red', label ='Real Stock Price')
plt.plot(date_list,forecast_close, color ='blue', label ='Predicted Stock Price')
plt.title(fd + "_LSTM")
plt.xlabel('Time')
plt.xticks(rotation = 30)
plt.ylabel('Stock Price')
plt.legend()
plt.savefig(plot_path + "_LSTM.png" ,bbox_inches="tight")
plt.show()

