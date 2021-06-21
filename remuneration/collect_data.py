import pandas as pd
import numpy as np
import os
from datetime import datetime
import pandas_datareader.data as web

start = datetime(2010, 10, 1)
end = datetime(2019, 12, 31)
stock_list = ["TWII","GSPC","HSI","N225","DJI"]
dir_path = 'D:/Time_Series_Research/no_date_data/'

for name in stock_list:
    print(name)
    data = web.DataReader("^" + name, 'yahoo', start, end)
    data.reset_index(inplace=True)
    data.dropna(axis=0, how='any', inplace=True)
    data = data.reset_index(drop=True)

    for j in range(1, 10):
        d1 = datetime(2010 + j, 1, 1, 0, 0)
        d2 = datetime(2010 + j, 12, 31, 0, 0)
        df = data.loc[(data['Date'] >= d1) & (data['Date'] <= d2)]
        df.reset_index(drop=True,inplace =True)
        new_fd = name + "_" + str(2010 + j) + ".csv"
        full_path = os.path.join("D:/Time_Series_Research/remuneration/origin_data", new_fd)
        df.to_csv(full_path, index=0, header=1)
        print('File:', full_path)