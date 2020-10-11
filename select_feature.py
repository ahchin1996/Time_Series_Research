import pandas as pd
import numpy as np

fd_path = 'D:/Time_Series_Research/new_data/GSPC/GSPC_2012.csv'
df = pd.read_csv(fd_path,sep=',',header=0)

str_in = input("Whic feature you want to choose？")

choose_feature = [int(n) for n in str_in.split()]

new_df = df[['Date','Close']]

for i in range(0,len(choose_feature)):
    new_df = pd.concat([new_df, df.iloc[:,choose_feature[i]]], axis=1)

new_df.columns