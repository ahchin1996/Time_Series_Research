import pandas as pd
import os

path =  'D:/Time_Series_Research/new_data/新增資料夾/TWII.csv'
df = pd.read_csv(path,sep=',',header=0)

df["Date"] = pd.to_datetime(df["Date"])
df.index = df.Date

def getLAG(price, period):
    lag = price.shift(period)
    return lag

# df[""] = getLAG(df.,1)

df["BIAS_6_L1"] = getLAG(df.BIAS_6,1)

df_year = df["2017"]

dir_path = 'D:/Time_Series_Research/new_data/TWII/'

full_path = os.path.join(dir_path,"TWII_2017.csv")
df_year.to_csv(full_path, index=0, header=1)
