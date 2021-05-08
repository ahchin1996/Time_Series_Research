import os
import pandas as pd
import numpy as np

path = 'D:/Time_Series_Research/new_data/ALL_DATA'
year_list = list(range(2011,2020))

def statistics_table(path,year_list):
    result = pd.DataFrame()
    for fd in os.listdir(path):
        full_path=os.path.join(path,fd)
        if os.path.isdir(full_path):
            print('Enter dir:',full_path)

            for fd_2 in os.listdir(full_path):
                full_path_2 = os.path.join(full_path, fd_2)
                if os.path.isdir(full_path_2):
                    continue
                else:
                    print('檔案:', full_path_2)

                    year = fd_2.strip(f'{fd}')
                    year = int(year.strip('_.csv'))
                    if int(year) in year_list:
                        lst = []
                        name = fd_2.strip(".csv")
                        df = pd.read_csv(full_path_2, sep=',', header=0)
                        close = df.Close
                        range = max(close) - min(close)
                        std = close.std()
                        Max = close.max()
                        Min = close.min()

                        lst.append(np.format_float_positional(range, precision=4))
                        lst.append(np.format_float_positional(std, precision=4))
                        lst.append(np.format_float_positional(Max, precision=4))
                        lst.append(np.format_float_positional(Min, precision=4))
                        lst = np.array(lst)
                        lst = pd.DataFrame(lst.reshape(1, 4), index=[name], columns=['Range', 'std', 'max', 'min'])
                        result = pd.concat([result, lst], axis=0)
                    else:
                        print("Not my choice year!\n")
        else:
            print('檔案:', full_path)
    return result

output = statistics_table(path,year_list)
output.to_csv(os.path.join(path, 'statistics_table.csv'), index=1, header=1)
