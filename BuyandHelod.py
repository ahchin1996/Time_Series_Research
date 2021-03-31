import backtrader as bt
from datetime import datetime

class BuyAndHold_Buy(bt.Strategy):
    def start(self):
        # set the starting cash
        self.val_start = self.broker.get_cash()
    def nextstart(self):
        # Buy stocks with all the available cash
        size = int(self.val_start / self.data)
        self.buy(size=size)
    def stop(self):
        # calculate the actual returns
        self.roi = (self.broker.get_value() / self.val_start) - 1.0
        print("ROI: %.2f, Cash: %.2f" % (100.0 * self.roi, self.broker.get_value()))

data = bt.feeds.YahooFinanceData(
    dataname="MSFT", fromdate=datetime(2010, 1, 1), todate=datetime(2020, 10, 23)
)

cerebro = bt.Cerebro()
cerebro.adddata(data)
cerebro.addstrategy(BuyAndHold_Buy, "HODL")
cerebro.broker.setcash(100000.0)
cerebro.run()

cerebro.plot(iplot=False)