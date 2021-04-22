import pandas as pd
import numpy as np
import os
from datetime import datetime
import pandas_datareader.data as web
import talib as ta

start = datetime(2010, 1, 1)
end = datetime(2019, 12, 31)
stock_list = ["TWII","GSPC","HSI","N225","DJI"]
dir_path = 'D:/Time_Series_Research/new_data/新增資料夾/'

for name in stock_list:
    data = web.DataReader("^" + name, 'yahoo', start, end)
    data.reset_index(inplace=True)
    data.dropna(axis=0, how='any', inplace=True)
    data = data.reset_index(drop=True)
    close = data['Close']
    day = len(close)

    # Moving average
    data['MA_5'] = ta.MA(close,5)
    data["MA_20"] = ta.MA(close, 20)

    # Bollinger Bands (Ma20)
    data["BBands_up"], middleband, data["BBands_down"] = ta.BBANDS(close,20 ,2,2,matype = 0)

    # Relative Strength Index
    def getRSI(Close, period=6):
        # 整理資料
        import pandas as pd
        Chg = Close - Close.shift(1)
        Chg_pos = pd.Series(index=Chg.index, data=Chg[Chg > 0])
        Chg_pos = Chg_pos.fillna(0)
        Chg_neg = pd.Series(index=Chg.index, data=-Chg[Chg < 0])
        Chg_neg = Chg_neg.fillna(0)

        # 計算12日平均漲跌幅度
        import numpy as np
        up_mean = []
        down_mean = []
        for i in range(period + 1, len(Chg_pos) + 1):
            up_mean.append(np.mean(Chg_pos.values[i - period:i]))
            down_mean.append(np.mean(Chg_neg.values[i - period:i]))

        # 計算 RSI
        rsi = []
        for i in range(len(up_mean)):
            rsi.append(100 * up_mean[i] / (up_mean[i] + down_mean[i]))
        rsi_series = pd.Series(index=Close.index[period:], data=rsi)
        return rsi_series

    data["RSI"] = getRSI(close,6)

    # Eeponential Moving Average
    data['EMA_12'] = ta.EMA(close,12)
    data['EMA_26'] = ta.EMA(close,26)

    # Moving Average Convergence / Divergence (MACD)
    data['DIF'], data['DEM'], data['OSC'] = ta.MACD(close,12, 26,9)

    # relative difference in percentage of price RDP
    data['RDP'] = ta.ROCP(close,1)*100

    # BIAS 乖離率
    def getBIAS(priceData, period):
        bias = 100 * (priceData - priceData.rolling(period).mean()) / priceData.rolling(period).mean()
        return bias

    data['BIAS_6'] = getBIAS(close, 6)
    data['BIAS_12'] = getBIAS(close, 12)
    data['BIAS_24'] = getBIAS(close, 24)

    # Psychological Line
    def getPSY(close, period):
        diff = close - close.shift(1)
        diff[diff > 0] = 1
        diff[diff < 0] = 0
        psy = diff.rolling(period).mean()
        return psy

    data['PSY_12'] = getPSY(close, 12)
    data['PSY_24'] = getPSY(close, 24)

    # Williams %R n = 14
    def getWR(priceData, period):
        array_highest = priceData.rolling(period).max()
        array_lowest = priceData.rolling(period).min()
        wr = np.zeros((len(priceData),))
        wr[:period] *= np.nan
        for i in range(period - 1, len(priceData)):
            wr[i] = (array_highest[i] - priceData[i]) / (array_highest[i] - array_lowest[i]) * 100
        return wr
    data['W%R'] = getWR(close, 14)

    # Stochastic K%D
    def getKD(priceData, period):
        array_highest = priceData.rolling(period).max()
        array_lowest = priceData.rolling(period).min()
        k = np.zeros((len(priceData),))
        k[:period] *= np.nan
        for i in range(period - 1, len(priceData)):
            k[i] = (close[i] - array_lowest[i]) / (array_highest[i] - array_lowest[i]) * 100
        k1 = pd.DataFrame(k)
        d = k1.rolling(3).mean()
        d = np.array(d)
        return k, d

    data['%K'], data['%D'] = getKD(close, 9)

    # PROC
    data['PROC'] = ta.ROCP(close,12)*100

    data['MO'] = ta.MOM(close,1)


    def getLAG(price, period):
        lag = price.shift(period)
        return lag

    data['LAG'] = getLAG(close, 1)

    data.drop(["Open", 'High', 'Low', 'Adj Close'], axis=1,inplace=True)

    new_fd = name + "_full.csv"
    full_path = os.path.join(dir_path, new_fd)
    data.to_csv(full_path, index=0, header=1)
    print('File:', full_path)

    print()
    print("---Next Stock---")
    print()

print("Process Completed!")