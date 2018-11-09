# -*- coding: utf-8 -*-

import pandas as pd



melbourne_file_path = '/home/pepe/DATOS/Kaggle/hausing_price/melbourne/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_data.describe()
melbourne_data.describe()['Rooms']
melbourne_data.columns
print(melbourne_data.dtypes)
print(melbourne_data.mean())

N = len(melbourne_data.columns)
for i in range(N):
    print(melbourne_data.columns[i])

for i in range(N):
    val = melbourne_data.columns[i]
    print(str(val))
    #print(melbourne_data.describe()[val])
try:
    print(melbourne_data.describe()['Price'])


