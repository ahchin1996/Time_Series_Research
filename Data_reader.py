import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame


start = datetime.datetime(1999, 1, 1)
end = datetime.datetime(2019, 12, 31)

df = web.DataReader("^TWII", 'yahoo', start, end)
df.head()
df.info()

def gettMA(priceData, period):
    ma = priceData.rolling(period).mean().fillna(0)
    return ma

df['MA_5'] = gettMA(df.Close, 5)
