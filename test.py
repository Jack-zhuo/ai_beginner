# [[  1.43151248]
#  [-28.37530069]]
import numpy as np

weights = np.array([
    [1.43151248],
    [-28.37530069]
])

features = np.array([
    [20, 1]
])

prediction = 1 / (1 + np.exp(-np.dot(features, weights)))
print(prediction)
