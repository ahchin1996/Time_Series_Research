import pandas as pd

# fd_path =  'D:/Time_Series_Research/new_data/DJI/DJI_2001.csv'
# df = pd.read_csv(fd_path, sep=',', header=0)
#
# fd = "DJI_2001.csv"
def chose_list(fd):
    fd = fd.strip(".csv")
    feature = {
        "DJI_2011": [8, 9, 11, 12, 14, 15, 23, 24], # 25 26
        "DJI_2012": [3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 23, 24], # 25 26
        "DJI_2013": [3, 4, 5, 6,  9, 11, 15, 16, 24], # 25 26
        "DJI_2014": [3, 4, 5,  8, 9, 10,  14, 15, 16, 24], # 25 26
        "DJI_2015": [3, 4, 5, 6, 8, 9, 12, 13, 14, 15, 16, 23, 24], # 25
        "DJI_2016": [3, 5, 6, 8, 9, 11, 16, 24], # 25
        "DJI_2017": [3, 4, 5, 6, 8, 9, 11, 12, 13, 23, 24], # 25 26 27
        "DJI_2018": [3,4,5,6,8,9,10,11,12,13,14,15,16,21,22,23,24], #25 26
        "DJI_2019": [3,4,5,6,8,9,10,12,13,14,15,16,23,24],   #  NA

        "GSPC_2011": [3,  8, 9, 10, 14, 15, 16, 23, 24], # 25 26
        "GSPC_2012": [3, 4, 5, 6, 8, 9, 11, 14, 16, 23, 24], # 25
        "GSPC_2013": [4, 5, 6, 8, 9, 10, 12, 23, 24], # 25 26 27
        "GSPC_2014": [3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 23, 24], # NA
        "GSPC_2015": [3, 4, 5, 8, 9, 10, 14, 15, 16, 23, 24], # 25
        "GSPC_2016": [3, 4, 5, 6, 8, 9,10, 14, 15, 16, 24], # 25
        "GSPC_2017": [3,4,5,6, 8, 9, 13,16, 24], # NA
        "GSPC_2018": [3,8,9,10,11,13,14,16,18,23,24], #25 26 27 28 29
        "GSPC_2019": [3,4,5,6,8,9,11,12,14,15,16,23,24], # 25 26 27 28

        "HSI_2011": [3, 4, 8, 9, 10, 12, 14, 16, 24], # 25
        "HSI_2012": [8, 9, 10, 11, 24], # NA
        "HSI_2013": [3, 5, 6, 8, 9, 11, 12, 14, 16, 23, 24], # 25 26
        "HSI_2014": [3, 4, 5, 6, 8, 9, 14, 15, 16, 23, 24], # 25
        "HSI_2015": [3, 5, 6, 8, 9, 10, 11, 12, 14, 16, 23, 24], # 25
        "HSI_2016": [4, 5, 6, 8, 9, 10, 12, 13, 15, 16, 23, 24], # 25 26
        "HSI_2017": [3, 4, 5, 6, 8, 9, 16, 24], # 25
        "HSI_2018": [3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 23, 24], # 25 26
        "HSI_2019": [3, 4, 5, 8, 9, 10, 13, 14, 15, 16, 23, 24], # 25

        "N225_2011": [3, 9, 24], # NA
        "N225_2012": [3, 4, 5, 9, 10, 23, 24], # 25 26
        "N225_2013": [3, 4, 8, 9, 10, 11, 14, 16, 23, 24], # 25
        "N225_2014": [3, 8, 9, 10, 16, 23, 24], # 25 26
        "N225_2015": [8, 11, 23, 24], # NA
        "N225_2016": [3, 4, 8, 9, 10, 14, 15, 16, 23, 24], # 25
        "N225_2017": [3, 4, 5, 6, 8, 9, 16, 24], # 25 26 27
        "N225_2018": [3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 19, 21, 23, 24], # 25 26
        "N225_2019": [3, 6, 8, 9, 10, 11, 14, 15, 16, 23, 24],  # 25 26

        "TWII_2011":[4, 5, 6, 8, 9, 16, 24],  # 25 26 27
        "TWII_2012":[3, 6, 8, 9, 12, 13, 14, 23, 24],  # 25
        "TWII_2013":[3, 4, 8, 9, 11, 16, 23, 24],  # 25
        "TWII_2014":[5, 6, 8, 9, 10, 12, 14, 23, 24],  # 25
        "TWII_2015":[3, 6, 8, 9, 10, 11, 23, 24],  # 25 26
        "TWII_2016":[3, 4, 5, 6, 8, 9, 11, 12, 14, 16, 24], # 25
        "TWII_2017":[3, 4, 8, 9, 14, 24], # 25
        "TWII_2018":[3, 4, 5, 8, 9, 13, 14, 15, 16, 23, 24], # 25 26
        "TWII_2019":[3, 4, 5, 6, 8, 9, 10, 11, 12, 15, 16, 23, 24] # 25
    }
    feature_list = feature[fd]
    # new_list = [int(n) for n in feature_list]
    return feature_list

# feature_list = chose_list(fd)
#
# new_df = df[['Date', 'Close']]
# for i in range(0, len(feature_list)):
#     new_df = pd.concat([new_df, df.iloc[:, feature_list[i]]], axis=1)

