import pandas as pd
import numpy as np
import os
from datetime import datetime

dir_path = 'D:/Time_Series_Research/new_data/'
for fd in os.listdir(dir_path):
    full_path = os.path.join(dir_path, fd)
    if os.path.isdir(full_path):
        continue
    else:
        data = pd.read_csv(full_path,sep=',',header=0)
        data = data.dropna(axis=0, how='any', inplace=False)
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
        rsi = []
        for i in range(6):
            rsi.append(0)

        for i in range(day - 6):
            negative = 0
            postive = 0
            r = 0
            price = []

            for j in range(7):
                price.append(close[i + j])

            for k in range(1, len(price)):
                n = 0
                n = price[k] - price[k - 1]
                if n > 0:
                    postive = postive + n
                else:
                    negative = negative + abs(n)
            r = ((postive / 6) / ((postive / 6) + (negative / 6))) * 100
            rsi.append(r)

        rsi = pd.DataFrame(rsi)
        data = pd.concat([data, rsi], axis=1)
        data = data.rename(columns={0: 'RSI'})

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

        # def getLAG(priceData, period):
        #     l = np.zeros((len(priceData),))
        #     l[:period] *= np.nan
        #     for i in range(period, len(priceData)):
        #         l[i] = priceData[i - period]
        #     return l
        #
        # data['LAG'] = getLAG(close,1)

        def getLAG(close , period):
            lag = close.shift(period)
            return lag

        data["LAG"] = getLAG(close,1)


        data = data.drop([ "Open", 'High', 'Low', 'Adj Close'], axis=1)

        dir_name = fd.strip("^.csv")
        
        new_path = os.path.join(dir_path, dir_name)
        os.mkdir(new_path)
        output_dor_path = new_path

        data['Date'] = pd.to_datetime(data['Date'] )
        for i in range(0,20):
            d1 = datetime(2000+i, 1, 1, 0, 0)
            d2 = datetime(2000+i, 12, 31, 0, 0)
            df = data[(data['Date'] > d1) & (data['Date'] < d2)]
            new_fd = fd.strip("^.csv") + "_" + str(2000+i) + ".csv"
            df.to_csv(os.path.join(output_dor_path, new_fd), index=0, header=1)
            full_path = os.path.join(output_dor_path, new_fd)
            print('檔案:', full_path)
    print()
    print("---Next Stock---")
    print()
