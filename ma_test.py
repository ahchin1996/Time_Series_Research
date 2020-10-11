# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:06:24 2020

@author: cooke
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
full_path = 'D:/Time_Series_Research/new_data/^HSI.csv'
data = pd.read_csv(full_path,sep=',',header=0)
data = data.dropna(axis=0, how='any', inplace=False)
data = data.reset_index(drop=True)
# data = pd.DataFrame(data)


close = data['Close']
day = len(close)



ma20 = []
BBands_up = []
BBands_down = []
for i in range(19):
    ma20.append(0)
    BBands_up.append(0)
    BBands_down.append(0)

for i in range(day - 19):
    BBands = []
    price = 0
    bu = 0
    bd = 0
    for j in range(20):
        price = price + close[i + j]
        BBands.append(close[i + j])
    BBands = pd.DataFrame(BBands)
    sd = BBands.std()
    price = BBands.mean()
    bu = price[0] + 2 * sd[0]
    bd = price[0] - 2 * sd[0]
    ma20.append(price[0])
    BBands_up.append(bu)
    BBands_down.append(bd)

ma20 = pd.DataFrame(ma20)
data = pd.concat([data, ma20], axis=1)
data = data.rename(columns={0: 'Ma20'})

BBands_up = pd.DataFrame(BBands_up)
data = pd.concat([data, BBands_up], axis=1)
data = data.rename(columns={0: 'BBands_up'})

BBands_down = pd.DataFrame(BBands_down)
data = pd.concat([data, BBands_down], axis=1)
data = data.rename(columns={0: 'BBands_down'})