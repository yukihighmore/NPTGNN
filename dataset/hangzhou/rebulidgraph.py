import pandas as pd
import numpy as np
import pickle

data = pd.read_csv('record_2019-01-01.csv')

A = data[data['lineID'] == 'A']
B = data[data['lineID'] == 'B']
C = data[data['lineID'] == 'C']

A_station = set(A['stationID'].tolist())
B_station = set(B['stationID'].tolist())
C_station = set(C['stationID'].tolist())

print(A_station) # {67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80} 4号线
print(B_station) # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33} 1号线
print(C_station) # {34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66} 2号线

# the similarity graph of metro
# the correlation graph of metro

with open('./graph_hz_sml.pkl', 'rb') as f:
    degree = pickle.load(f, encoding='ISO-8859-1')
print(degree)
num = len(degree)
degree[degree > 0.4] = 1

f = open('sml.csv', 'w')
for i in range(num):
    for j in range(i, num):
        if i != j:
            if degree[i, j] == 1:
                f.write(str(i) + ', ' + str(j) + '\n')
f.close()