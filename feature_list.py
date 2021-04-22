import pandas as pd

# fd_path =  'D:/Time_Series_Research/new_data/DJI/DJI_2001.csv'
# df = pd.read_csv(fd_path, sep=',', header=0)
#
# fd = "DJI_2001.csv"
def chose_list(fd):
    fd = fd.strip(".csv")
    feature = {
        "DJI_2011":[8,9,11,12,14,15,23,24],
        "DJI_2017":[3,4,5,6,8,9,11,12,13,23,24],
        "DJI_2018":[3,4,5,6,8,9,10,11,12,13,14,15,16,21,22,23,24],
        "DJI_2019":[3,4,5,6,8,9,10,12,13,14,15,16,23,24],
        "GSPC_2018":[3,8,9,10,11,13,14,16,18,23,24],
        "GSPC_2019":[3,4,5,6,8,9,11,12,14,15,16,23,24],
        "HSI_2018":[3,4,5,6,8,9,10,12,14,15,16,23,24],
        "HSI_2019":[3,4,5,8,9,10,13,14,15,16,23,24],
        "N225_2018":[3,4,5,6,8,9,11,12,13,14,15,16,19,21,23,24],
        "N225_2019":[3,6,8,9,10,11,14,15,16,23,24],
        "TWII_2018":[3,4,5,8,9,13,14,15,16,23,24],
        "TWII_2019":[3,4,5,6,8,9,10,11,12,15,16,23,24]
    }
    feature_list = feature[fd]
    # new_list = [int(n) for n in feature_list]
    return feature_list

# feature_list = chose_list(fd)
#
# new_df = df[['Date', 'Close']]
# for i in range(0, len(feature_list)):
#     new_df = pd.concat([new_df, df.iloc[:, feature_list[i]]], axis=1)

