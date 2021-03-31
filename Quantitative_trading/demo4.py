import datetime
import backtrader as bt
import pandas as pd
import backtrader.feeds as btfeeds
import pandas_datareader as pdr
from tiingo import TiingoClient
import time

config = {}
config['session'] = True
config['api_key'] = "63e3bed28936dbb6416aeb18440310f60884fcf6"
client = TiingoClient(config)
historical_prices = client.get_ticker_price("GOOGL",
                                            fmt='json',
                                            startDate='2019-01-01',
                                            endDate='2019-06-30',
                                            frequency='daily')

class DT_Line(bt.Indicator):
    lines = ('U','D')
    params = (('period',2),('k_u',.7),('k_d',.7))
    def __init__(self):
        self.addminperiod(self.p.period +1)
        # self.plotinfo.plotmaster = self.data

    def next(self):
        HH = max(self.data.high.get(-1,size = self.p.period))
        LC = min(self.data.close.get(-1,size = self.p.period))
        HC = max(self.data.close.get(-1,size = self.p.period))
        LL = min(self.data.low.get(-1,size = self.p.period))
        r = max(HH - LC  ,LL - HC)
        self.lines.U[0] = self.data.open[0] + self.p.k_u * r
        self.lines.D[0] = self.data.open[0] - self.p.k_d * r

class DualThrust(bt.Strategy):
    def __init__(self):
        self.dataclose = self.data0.close
        self.D_Line = DT_Line(self.data1)
        self.D_Line = self.D_Line() #映射到相同時間維度上
        self.D_Line.plotinfo.plotmaster = self.data0
        # self.D_Line.plotinfo.plot = False
        self.buy_sig = bt.indicators.CrossOver(self.dataclose,self.D_Line.U)
        self.sell_sig = bt.indicators.CrossDown(self.dataclose,self.D_Line.D)

    def start(self):
        print("the world call me")

    def prenext(self):
        print("not mature")

    def next(self):
        if self.data.datetime.time() > datetime.time(9,3) and self.data.datatime.time < datetime.time(15,30):
            if not self.position and self.buy_sig[0] ==1:
                self.order = self.buy()
            if not self.position and self.sell_sig[0] == 1:
                self.order = self.sell()
            if self.getposition().size < 0 and self.buy_sig[0] == 1:
                self.order = self.close()
                self.order = self.buy()
            if self.getposition().size > 0 and self.sell_sig[0] ==1:
                self.order  = self.close()
                self.order = self.sell()

        if self.data.datetime.time() > datetime.time(15,30) and self.position:
            self.order = self.close()

if __name__ == "__main__":

    cerebro = bt.Cerebro()

    TWII = pdr.data.DataReader("AAPL", 'yahoo')

    brf_min_bar = bt.feeds.PandasData(
    dataname= TWII,
    fromdate=datetime.datetime(2019, 1, 25),
    todate=datetime.datetime(2019, 6, 26),
    timeframe=bt.TimeFrame.Minutes
    )

    cerebro.adddata(brf_min_bar)
    cerebro.resampledata(brf_min_bar,timeframe = bt.TimeFrame.Days)

    cerebro.addstrategy(DualThrust)

    cerebro.run()

    cerebro.plot(style = "candle")