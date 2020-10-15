import pandas as pd
feature = {
    "DJI_2001":[8,9,15,16,23,24],
    "DJI_2003":[8,9,23,24]
}

fd_path =  'D:/Time_Series_Research/new_data/DJI/DJI_2001.csv'
df = pd.read_csv(fd_path, sep=',', header=0)

fd = "DJI_2001.csv"
fd = fd.strip(".csv")

str_in = input("Whic feature you want to choose？")

if str_in.lower() == "all":
    new_df = df
else:
    feature_list = feature[fd]
    choose_feature = [int(n) for n in feature_list]
    new_df = df[['Date', 'Close']]
    for i in range(0,len(choose_feature)):
        new_df = pd.concat([new_df, df.iloc[:, choose_feature[i]]], axis=1)


print(new_df.columns)
