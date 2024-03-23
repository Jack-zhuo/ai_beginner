import numpy as np
import collections as c

data = np.loadtxt('data0.csv', delimiter=',')

predict_point = 200

feature = data[:, 0]
target = data[:, 1]

distance = np.abs(feature - predict_point)
print(feature)
print(target)
index = np.argsort(distance)
target_sort = target[index]

k = 36
target_sort_k = target_sort[:k]
res = c.Counter(target_sort_k).most_common()[0][0]
print(res)