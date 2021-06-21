import os
import pandas as pd
import numpy as np
import talib as ta
from datetime import datetime
import pandas_datareader.data as web
import pandas_datareader as pdr

SPY = pdr.get_data_tiingo('SPY', api_key='6af47abd76fbc371d5606fca0694502b866c7bcf')
SPY = SPY.reset_index(level=[0,1])
SPY.index = SPY['date']
SPY_adj = SPY.iloc[:,7:12]
SPY_adj.columns = ['Close','High','Low','Open','Volume']

# 取開盤價、收盤價
close = SPY_adj.Close
Open = SPY_adj.Open
Open2019 = Open['2019']

ma5 = ta.SMA(close, timeperiod=5)
ma10 = ta.SMA(close, timeperiod=10)

# 5ma與10ma差距
MA_dif = ma5 - ma10
MA_dif = MA_dif['2019']

stock = 0
sig = [0]

for i in range(1,len(MA_dif)):
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
ma_sig = pd.Series(index = MA_dif.index, data = sig)
ma_sig_2019 = ma_sig['2019']

# 每次買賣的報酬率
rets = []
transaction = []

# 是否仍有庫存
stock = 0
stock_his = []

# 當次交易買入價格
buy_price = 0

# 當次交易賣出價格
sell_price = 0

# 每次買賣的報酬率
for i in range(len(ma_sig_2019)-1):
    stock_his.append(stock)
    if ma_sig_2019[i] == 1:
        # 隔日開盤買入
        buy_price = Open2019[ma_sig_2019.index[i+1]]
        stock += 1
        # 紀錄交易日期
        transaction.append([ma_sig_2019.index[i+1],'buy'])
    elif ma_sig_2019[i] == -1:
        # 隔日開盤賣出
        sell_price = Open2019[ma_sig_2019.index[i+1]]
        stock -= 1
        rets.append((sell_price-buy_price)/buy_price)
        # 賣出後就清空資料
        buy_price = 0
        sell_price = 0
        # 紀錄交易日期
        transaction.append([ma_sig_2019.index[i+1],'sell'])

# 如果最後手上有庫存，就用回測區間最後一天的開盤價賣掉
if stock == 1 and buy_price != 0 and sell_price == 0:
    sell_price = Open2019[-1]
    rets.append((sell_price-buy_price)/buy_price)
    stock -= 1
    transaction.append([Open2019.index[-1],'sell'])

# 總報酬率
total_ret = 1
for ret in rets:
    total_ret *= 1 + ret
# print(str(round((total_ret - 1)*100,2)) + '%')
print('總報酬率：' + str(round(100*(total_ret-1),2)) + '%')
transaction