import numpy as np

arr = np.random.randint(-127,127,(100,5), dtype=int)

arr[:,4] = 0
result = ""
for row in arr:
    #row[4] = row[0]*-5+row[1]*10+row[2]*27+row[3]*-13+1*7
    result += np.binary_repr(row[0], width=8) + '_'  + np.binary_repr(row[1], width=8) + '_'  + np.binary_repr(row[2], width=8) + '_'  + np.binary_repr(row[3], width=8) + '_' + np.binary_repr(row[0]*-5+row[1]*10+row[2]*27+row[3]*-13+1*7, width=16) + '\n'

#bin_arr = [np.binary_repr(x, width=8) for x in arr]

print(result)