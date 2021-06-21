import pandas as pd
import numpy as np
import os
from datetime import datetime
import pandas_datareader.data as web
import talib as ta
import datetime


SPY_adj = web.DataReader("SPY" , 'yahoo')

# 取開盤價、收盤價
close = SPY_adj.Close
Open = SPY_adj.Open
Open2019 = Open['2019']

# 計算均線
import talib
ma5 = talib.SMA(close, timeperiod=5)
ma10 = talib.SMA(close, timeperiod=10)

# 繪圖
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
plt.plot(ma5['2019'])
plt.plot(ma10['2019'])
plt.legend(['5ma','10ma'])
plt.xticks(rotation='vertical')

# 5ma與10ma差距
MA_dif = ma5 - ma10
MA_dif = MA_dif['2019']

# 參數
stock = 0
sig = []

# 訊號
for i in range(len(MA_dif)):
    # 5MA往上穿越10MA
    if MA_dif[i - 1] < 0 and MA_dif[i] > 0 and stock == 0:
        stock += 1
        sig.append(1)

    # 5MA往下穿越10MA
    elif MA_dif[i - 1] > 0 and MA_dif[i] < 0 and stock == 1:
        stock -= 1
        sig.append(-1)
    else:
        sig.append(0)

# 製作dataframe
import pandas as pd

ma_sig = pd.Series(index=MA_dif.index, data=sig)
ma_sig_2019 = ma_sig['2019']