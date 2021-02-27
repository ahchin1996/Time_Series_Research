import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame

#
# start = datetime.datetime(1999, 1, 1)
# end = datetime.datetime(2019, 12, 31)
#
# df = web.DataReader("^TWII", 'yahoo', start, end)
# df.head()
# df.info()

from feature_list import chose_list

fd_path =  'D:/Time_Series_Research/new_data/DJI/DJI_2001.csv'
df = pd.read_csv(fd_path, sep=',', header=0)

fd = "DJI_2001.csv"

feature_list = chose_list(fd)

new_df = df[['Date', 'Close']]
for i in range(0, len(feature_list)):
    new_df = pd.concat([new_df, df.iloc[:, feature_list[i]]], axis=1)

new_df.columns



start = datetime.datetime(2020, 1, 2)
end = datetime.datetime(2020, 12, 31)

df = web.DataReader("^TWII", 'yahoo', start, end)
df.head()
df.info()
df.insert(0,"Date",df.index)

df.reset_index()

output_path ='D:/Time_Series_Research/new_data/TWII_2020.csv'
df.to_csv(output_path, index=1, header=1)