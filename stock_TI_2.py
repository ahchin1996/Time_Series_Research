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
    close = data['Close']
    day = len(close)

    # Ma5
    def gettMA(priceData, period):
        ma = priceData.rolling(period).mean().fillna(0)
        return ma

    data['MA_5'] = gettMA(close, 5)

    # Bollinger Bands (Ma20)
    data["MA_20"] = gettMA(data.Close, 20)
    data["BBands_up"] = data.MA_20 + data.Close.rolling(20).std().fillna(0) * 2
    data["BBands_down"] = data.MA_20 - data.Close.rolling(20).std().fillna(0) * 2

    # RSI Relative Strength Index
    # rsi = []
    # for i in range(6):
    #     rsi.append(0)
    #
    # for i in range(day - 6):
    #     negative = 0
    #     postive = 0
    #     r = 0
    #     price = []
    #
    #     for j in range(7):
    #         price.append(close[i + j])
    #
    #     for k in range(1, len(price)):
    #         n = 0
    #         n = price[k] - price[k - 1]
    #         if n > 0:
    #             postive = postive + n
    #         else:
    #             negative = negative + abs(n)
    #     r = ((postive / 6) / ((postive / 6) + (negative / 6))) * 100
    #     rsi.append(r)
    #
    # rsi = pd.DataFrame(rsi)
    # data = pd.concat([data, rsi], axis=1)
    # data = data.rename(columns={0: 'RSI'})
    # RSI函數
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


    # 稍微對照一下剛剛算出來的數字，會是一樣的
    data["RSI"] = getRSI(close,6)

    # Eeponential Moving Average EMA
    data['EMA_12'] = data['Close'].ewm(span=12).mean()
    data['EMA_26'] = data['Close'].ewm(span=26).mean()

    # Moving Average Convergence / Divergence (MACD)
    # DIF
    ema12 = data['EMA_12']
    ema26 = data['EMA_26']
    dif = []
    for i in range(day):
        p = ema12[i] - ema26[i]
        dif.append(p)
    data['DIF'] = dif

    # DEM (MACD)
    data['DEM'] = data['DIF'].ewm(span=9).mean()

    # OSC = DIF - DEM
    data['OSC'] = data['DIF'] - data['DEM']

    # relative difference in percentage of price RDP
    rdp = []
    rdp.append(0)
    for i in range(1, day):
        n = (close[i] - close[i - 1]) / close[i - 1] * 100
        rdp.append(n)
    rdp = pd.DataFrame(rdp)
    data = pd.concat([data, rdp], axis=1)
    data = data.rename(columns={0: 'RDP'})

    # BIAS
    def getBIAS(priceData, period):
        bias = 100 * (priceData - priceData.rolling(period).mean()) / priceData.rolling(period).mean()
        return bias

    data['BIAS_6'] = getBIAS(close, 6)
    data['BIAS_12'] = getBIAS(close, 12)
    data['BIAS_24'] = getBIAS(close, 24)

    # PSY
    def getPSY(priceData, period):
        difference = priceData[1:].reset_index(drop=True) - priceData[:-1]
        difference = np.append(0, difference)
        difference_dir = np.where(difference > 0, 1, 0)
        psy = np.zeros((len(priceData),))
        psy[:period] *= np.nan
        for i in range(period, len(priceData)):
            psy[i] = (difference_dir[i - period + 1:i + 1].sum()) / period
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
    def getROC(priceData, period):
        roc = np.zeros((len(priceData, )))
        roc[:period] *= np.nan
        x = period
        while x < len(priceData):
            roc[x] = (priceData[x] - priceData[x - period]) / priceData[x - period] * 100
            x += 1
        return roc

    data['PROC'] = getROC(close, 12)

    def getMomentum(priceData, period):
        m = np.zeros((len(priceData),))
        m[:period] *= np.nan
        for i in range(period, len(priceData)):
            m[i] = (priceData[i] - priceData[i - period]) / period
        return m

    data['MO'] = getMomentum(close, 1)


    def getLAG(price, period):
        lag = price.shift(period)
        return lag

    data['LAG'] = getLAG(close, 1)

    data.drop(["Open", 'High', 'Low', 'Adj Close'], axis=1,inplace=True)
    new_path = os.path.join(dir_path, name)
    os.mkdir(new_path)
    output_dir_path = new_path
    data['Date'] = pd.to_datetime(data['Date'])

    for j in range(1, 10):
        d1 = datetime(2010 + j, 1, 1, 0, 0)
        d2 = datetime(2010 + j, 12, 31, 0, 0)
        df = data.loc[(data['Date'] >= d1) & (data['Date'] <= d2)]
        df.reset_index(drop=True,inplace =True)
        df = df.drop(["Date"],axis=1)
        # id = pd.DataFrame(range(1, len(df)+1))
        # df.insert(0, "index", id)
        new_fd = name + "_" + str(2010 + j) + ".csv"
        full_path = os.path.join(output_dir_path, new_fd)
        df.to_csv(full_path, index=0, header=1)
        print('File:', full_path)

    print()
    print("---Next Stock---")
    print()

print("Process Completed!")