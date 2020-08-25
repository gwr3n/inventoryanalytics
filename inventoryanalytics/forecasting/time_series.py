import pandas as pd

data = [26664.4, 26828.5, 27201.5, 27387.0, 27433.5, 27791.4, 27686.9, 27976.8, 27896.7, 27931.0 ]
index= pd.date_range(start='2020-08-03', end='2020-08-14', freq='B')
# index = [pd.Timestamp('2020-08-03'), pd.Timestamp('2020-08-04'), pd.Timestamp('2020-08-05'),
#          pd.Timestamp('2020-08-06'), pd.Timestamp('2020-08-07'), pd.Timestamp('2020-08-10'),
#          pd.Timestamp('2020-08-11'), pd.Timestamp('2020-08-12'), pd.Timestamp('2020-08-13'),
#          pd.Timestamp('2020-08-14')] # alternative to date_range
dowdata = pd.Series(data, index)
print(dowdata)