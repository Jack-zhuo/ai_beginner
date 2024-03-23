import numpy as np

data = np.array([
    [80, 200],
    [95, 230],
    [104, 245],
    [112, 274],
    [125, 259],
    [135, 262]
])

feature = data[:, :1]
label = data[:, -1:]
m = 1
b = 1

weight = np.array([
    [m],
    [b]
])

feature_matrix = np.append(feature, np.ones(shape=(6, 1)), axis=1)
differences = np.dot(feature_matrix, weight) - label
gradient = np.dot(feature_matrix.T, differences)
print(gradient)
