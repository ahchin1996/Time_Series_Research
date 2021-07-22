import pandas as pd
import numpy as np
import os
from datetime import datetime

def Profitability(forecast_path,origin_path):
    dirname, basename = os.path.split(origin_path)
    root, extension = os.path.splitext(origin_path)
    stockName = dirname.split("/")[-1]
    year = int(basename.strip(extension)[-4:])
    fd = basename.strip(extension)

    forecast_close = pd.read_csv(forecast_path, sep = ",",names = ["Close"])
    origin_data = pd.read_csv(origin_path,sep=',',header=0)

    date_array = pd.to_datetime(origin_data['Date'])
    origin_data.set_index(origin_data.Date,inplace=True)

    openPrice = pd.Series(index= origin_data.index,data=origin_data.Open)

    split_no = 0
    while date_array.iloc[split_no] < datetime(year, 11,1, 0, 0):
        split_no +=1
    print(split_no)

    openPrice = openPrice.iloc[split_no:]
    origin_close = origin_data['Close'][split_no:]

    forecast_close.set_index(date_array[split_no:],inplace=True)
    forecast_close = forecast_close.squeeze()


    stock = 0
    sig = []

    for i in range(0,len(origin_close)-1):
        #買入訊號
        if (forecast_close[i + 1] - origin_close[i]) > 0 and stock == 0:
            stock += 1
            sig.append(1)
        #賣出訊號
        elif (forecast_close[i + 1] - origin_close[i]) < 0  and stock == 1:
            stock -= 1
            sig.append(-1)
        else:
            sig.append(0)
    sig.append(0)

    # 製作dataframe
    sig = pd.Series(index = openPrice.index, data = sig)

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

    #利潤
    profit = 0

    # 每次買賣的報酬率
    for i in range(len(sig)):
        stock_his.append(stock)
        if sig[i] == 1 :
            # 隔日開盤買入
            buy_price = openPrice[sig.index[i+1]]
            stock += 1
            # 紀錄交易日期
            transaction.append([sig.index[i+1],'buy'])
        elif sig[i] == -1 and stock == 1:
            # 隔日開盤賣出
            sell_price = openPrice[sig.index[i+1]]
            stock -= 1
            rets.append((sell_price-buy_price)/buy_price)

            profit = profit + (sell_price-buy_price)

            # 賣出後就清空資料
            buy_price = 0
            sell_price = 0
            # 紀錄交易日期
            transaction.append([sig.index[i+1],'sell'])

    # 如果最後手上有庫存，就用回測區間最後一天的開盤價賣掉
    if stock == 1 and buy_price != 0 and sell_price == 0:
        sell_price = openPrice[len(sig)-1]
        rets.append((sell_price-buy_price)/buy_price)
        stock -= 1
        profit = profit + (sell_price - buy_price)
        transaction.append([openPrice.index[-1],'sell'])

    # 總報酬率
    total_ret = 1
    for ret in rets:
        total_ret *= 1 + ret
    # print(str(round((total_ret - 1)*100,2)) + '%')
    print('總報酬率：' + str(round(100*(total_ret-1),2)) + '%')
    transaction
    return profit

forecast_path_dir = "D:/Time_Series_Research/remuneration"
origin_path_dir = "D:/Time_Series_Research/remuneration/origin_data"

forecast_list = os.listdir(forecast_path_dir)
origin_list = os.listdir(origin_path_dir)

pro = []
name = []

for o in origin_list:
    sublist = []
    o = o.strip(".csv")
    for s in forecast_list:
        if o in s:
            sublist.append(s)

    o = o + '.csv'
    for x in sublist:
        forecast_path = ""
        origin_path = ""
        forecast_path = os.path.join(forecast_path_dir, x)
        origin_path = os.path.join(origin_path_dir, o)

        pro.append(Profitability(forecast_path, origin_path))
        name.append(x)

        print(x)
        print(forecast_path)
        print(origin_path)

pro_list = pd.DataFrame(pro)
name_list = pd.DataFrame(name)
result = pd.concat([name_list,pro_list],axis=1)

# result.to_csv( "D:/Time_Series_Research/remuneration/A_pro.csv",index=0,header=0)
