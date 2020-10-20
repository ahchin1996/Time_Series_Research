import pandas as pd

# fd_path =  'D:/Time_Series_Research/new_data/DJI/DJI_2001.csv'
# df = pd.read_csv(fd_path, sep=',', header=0)
#
# fd = "DJI_2001.csv"
def chose_list(fd):
    fd = fd.strip(".csv")
    feature = {
        "DJI_2001":[8,9,15,16,23,24],
        "DJI_2003":[8,9,23,24],
        "DJI_2004":[8,15,23,24],
        "DJI_2008":[6,11,23,24],
        "GSPC_2001":[8,9,23,24],
        "GSPC_2003":[4,5,8,9,23,24],
        "GSPC_2004":[3,8,23,24],
        "GSPC_2008":[4,8,9,23,24],
        "HSI_2001":[8,9,23,24],
        "HSI_2003":[8,23,24],
        "HSI_2004":[23,24],
        "HSI_2008":[23,24],
        "N225_2001":[8,23,24],
        "N225_2003":[23,24],
        "N225_2004":[8,16,23,24],
        "N225_2008":[4,5,9,23,24],
        "TWII_2001":[8,23,24],
        "TWII_2003":[3,5,6,8,9,11,12,23,24],
        "TWII_2004":[4,5,6,9,12,23,24],
        "TWII_2008":[8,9,23,24]
    }
    feature_list = feature[fd]
    # new_list = [int(n) for n in feature_list]
    return feature_list

# feature_list = chose_list(fd)
#
# new_df = df[['Date', 'Close']]
# for i in range(0, len(feature_list)):
#     new_df = pd.concat([new_df, df.iloc[:, feature_list[i]]], axis=1)

