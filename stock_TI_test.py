import pandas as pd
import numpy as np
# import os
import matplotlib.pyplot as plt
from datetime import datetime 

test_dir = 'D:/Time_Series_Research/new_data/^DJI.csv'
data = pd.read_csv(test_dir,sep=",",header=0)
close = data['Close']
day = len(close)
data['Date'] = pd.to_datetime(data['Date'] )

#Ma5

def gettMA(priceData, period):
    ma = priceData.rolling(period).mean().fillna(0)
    return ma

data['MA_5'] = gettMA(data.Close, 5)
data["MA_20"] = gettMA(data.Close,20)
data["B_up"] = data.MA_20 + data.Close.rolling(20).std().fillna(0) * 2
data["B_down"] = data.MA_20 - data.Close.rolling(20).std().fillna(0) * 2
#布林通道Ma20
# data['MA_20'] = gettMA(close,20)

# ma20 = []
# BBands_up = []
# BBands_down = []
# for i in range(19):
#     ma20.append(None)
#     BBands_up.append(None)
#     BBands_down.append(None)
#
# for i in range(day-19):
#     BBands = []
#     price = 0
#     bu = 0
#     bd = 0
#     for j in range(20):
#         price = price + close[i+j]
#         BBands.append(close[i+j])
#
#     BBands = pd.DataFrame(BBands)
#     sd = BBands.std()
#     price = BBands.mean()
#     bu = price[0] + 2*sd[0]
#     bd = price[0] - 2*sd[0]
#     ma20.append(price[0])
#     BBands_up.append(bu)
#     BBands_down.append(bd)
#
# ma20 = pd.DataFrame(ma20)
# data = pd.concat([data,ma20],axis = 1)
# data = data.rename(columns = {0:'MA_20'})
#
# BBands_up = pd.DataFrame(BBands_up)
# data = pd.concat([data,BBands_up],axis = 1)
# data = data.rename(columns = {0:'BBands_up'})
#
# BBands_down = pd.DataFrame(BBands_down)
# data = pd.concat([data,BBands_down],axis = 1)
# data = data.rename(columns = {0:'BBands_down'})

d1 = datetime(2000, 1, 1, 0, 0)
d2 = datetime(2000, 12, 31, 0, 0)
df = data[(data['Date'] > d1) & (data['Date'] < d2)]

flg ,ax = plt.subplots(1,1)
plt.plot(df["Date"], df['B_up'],label = 'B_up')
plt.plot(df["Date"], df['MA_20'],color = 'red', label = 'MA_20')
plt.plot(df["Date"], df['B_down'],label = 'B_down')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.xticks(rotation = 30)
plt.ylabel('Stock Price')
plt.legend(loc = 'best')
plt.show()

flg ,ax = plt.subplots(1,1)
plt.plot(df["Date"], df['BBands_up'],label = 'B_up')
plt.plot(df["Date"], df['MA_20'],color = 'red', label = 'MA_20')
plt.plot(df["Date"], df['BBands_down'],label = 'B_down')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.xticks(rotation = 30)
plt.ylabel('Stock Price')
plt.legend(loc = 'best')
plt.show()

#RSI
rsi = []
for i in range(6):
    rsi.append(None)
    
for i in range(day-6):
    negative = 0
    postive = 0
    r = 0
    price = []
    
    
    for j in range(7):
        price.append(close[i+j])
    
    for k in range(1,len(price)):
        n = 0
        n = price[k]-price[k-1]
        if n > 0 :
            postive = postive + n
        else:
            negative = negative + abs(n)
    r = ((postive/6)/((postive/6)+(negative/6)))*100
    rsi.append(r)
    
rsi = pd.DataFrame(rsi)
data = pd.concat([data,rsi],axis = 1)
data = data.rename(columns = {0:'RSI'})

# Eeponential Moving Average EMA
data['EMA_12'] = data['Close'].ewm(span=12).mean()
data['EMA_26'] = data['Close'].ewm(span=26).mean()

#Moving Average Convergence / Divergence (MACD)
#DIF
ema12 = data['EMA_12']
ema26 = data['EMA_26']
dif = []
for i in range(day):
    p = ema12[i]-ema26[i]
    dif.append(p)
data['DIF'] = dif

# DEM (MACD)
data['DEM'] = data['DIF'].ewm(span=9).mean()

#OSC = DIF - DEM
data['OSC'] = data['DIF'] - data['DEM']

# relative difference in percentage of price RDP
rdp =[]
rdp.append(None)
for i in range(1,day):
    n = (close[i] - close[i - 1]) / close[i - 1]*100
    rdp.append(n)
rdp = pd.DataFrame(rdp)
data = pd.concat([data,rdp],axis = 1)
data = data.rename(columns = {0:'RDP'})

#BIAS
def getBIAS(priceData,period):
    bias = 100*(priceData-priceData.rolling(period).mean())/priceData.rolling(period).mean()
    return bias

data['BIAS6'] = getBIAS(close,6)
data['BIAS12'] = getBIAS(close,12)
data['BIAS24'] = getBIAS(close,24)

#PSY
def getPSY(priceData, period):
    difference = priceData[1:].reset_index(drop = True) - priceData[:-1]
    difference = np.append(0, difference)    
    difference_dir = np.where(difference > 0, 1, 0)
    psy = np.zeros((len(priceData),))
    psy[:period] *= np.nan
    for i in range(period, len(priceData)):
       psy[i] = (difference_dir[i-period+1:i+1].sum()) / period
    return psy

data['psy12'] = getPSY(close,12)
data['psy24'] = getPSY(close,24)


data.index = pd.to_datetime(data['Date'])

fig,ax = plt.subplots(1,1,figsize=(10,5))
data['psy12'].plot(ax=ax)
data['psy24'].plot(ax=ax)
ax.legend(loc = 'upper left')
ax.set_title('PSY')

# Williams %R n = 14
def getWR(priceData,period):
    array_highest = priceData.rolling(period).max()
    array_lowest = priceData.rolling(period).min()
    wr = np.zeros((len(priceData),))
    wr[:period] *= np.nan
    for i in range(period-1,len(priceData)):
        wr[i] = (array_highest[i] - priceData[i])/(array_highest[i] - array_lowest[i])*100
    return wr

data['W%R'] = getWR(close,14)

# Stochastic K%D
def getKD(priceData,period):
    array_highest = priceData.rolling(period).max()
    array_lowest = priceData.rolling(period).min()
    k = np.zeros((len(priceData),))
    k[:period] *= np.nan
    for i in range(period-1,len(priceData)):
        k[i] = (close[i] - array_lowest[i])/(array_highest[i]-array_lowest[i])*100
    k1 = pd.DataFrame(k)
    d = k1.rolling(3).mean()
    d = np.array(d)
    return k,d

data['%K'],data['%D'] = getKD(close,9)

fig,ax = plt.subplots(1,1,figsize=(10,5))
data['%K'].plot(ax=ax)
data['%D'].plot(ax=ax)
ax.legend(loc = 'upper left')
ax.set_title('K%D')


#PROC
def getROC(priceData,period):
    roc = np.zeros((len(priceData,)))
    roc[:period] *= np.nan
    x = period
    while x < len(priceData):
        roc[x] = (priceData[x] - priceData[x-period]) / priceData[x-period] *100
        x +=1
    return roc

data['PROC'] = getROC(close,12)

def getMomentum(priceData,period):
    m = np.zeros((len(priceData),))
    m[:period] *= np.nan
    for i in range(period,len(priceData)):
        m[i] = (priceData[i] - priceData[i - period])/period
    return m

data['MO'] = getMomentum(close,1)

def getLAG(priceData,period):
    m = np.zeros((len(priceData),))
    m[:period] *= np.nan
    for i in range(period,len(priceData)):
        m[i] = priceData[i-period]
    return m

data['LAG'] = getLAG(close,1)



# plot
fig,ax = plt.subplots(3,1,figsize=(10,10))
plt.subplots_adjust(hspace=0.5)
# ax[0].plot(data['MA_20'],label = 'BB')
data['MA_20'].plot(ax=ax[0],label = 'BB')
data['BBands_up'].plot(ax=ax[0])
data['BBands_down'].plot(ax=ax[0])
data['Close'].plot(ax=ax[0])
prop0 = {'title':'Bollinger Bands',
         'xlabel':'Time'}
ax[0].set(**prop0)
# ax[0].set_xlabel('Time')
# ax[0].set_xticks([0,63,126,189,252])
# ax[0].set_xticklabels(['2011.1','2011.4','2011.7','2011.10','2011.12'],rotation = 20,fontsize = 'small')
ax[0].legend(loc = 'upper left')

data['BB'].plot(ax=ax[1])
data['EMA_26'].plot(ax=ax[1])
data['Close'].plot(ax=ax[1])
ax[1].set_xlabel('Time')
ax[1].set_title('EMA')
ax[1].legend(loc = 'upper left')

data['DIF'].plot(ax=ax[2])
data['DEM'].plot(ax=ax[2])
ax[2].fill_between(data.index,0,data['OSC'])
ax[2].set_xlabel('Time')
ax[2].set_title('MACD')
ax[2].legend(loc = 'upper left')
plt.show()



# =============================================================================
# for fd in os.listdir(dir_path):
#     full_path = os.path.join(dir_path, fd)
#     if os.path.isdir(full_path):
#         continue
#     else:
#         data = pd.read_csv(full_path,sep=',',header=None)
#         columes_name = data.columns
#
#         data[columes_name[-1]] = data[columes_name[-1]].replace(1,'A')
#         data[columes_name[-1]] = data[columes_name[-1]].replace(2,'B')
#         data = data.fillna(0)
#
#         data.to_csv(os.path.join(output_dor_path, fd), index=0, header=0)
#
#         print('檔案:', full_path)
# =============================================================================


# data.drop(data.index[0:3])
# data.reset_index()
# df[df.index == 2]

# d1 = datetime(2011, 1, 1, 0, 0)
# d2 = datetime(2011, 12, 31, 0, 0)
# data['Date'] = pd.to_datetime(data['Date'] )
# df = data[(data['Date'] > d1) & (data['Date'] < d2)]
fd = "DJI"
a = "^DJI_2001.csv"
a = int(a.strip(f'^{fd}_.csv'))

# df.isnull().sum()
# df.describe()
# df.info()



