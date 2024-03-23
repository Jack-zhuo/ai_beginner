import numpy as np

data = np.array([
    [5, 0],
    [15, 0],
    [25, 1],
    [35, 1],
    [45, 1],
    [55, 1],
])

features = data[:, 0:1]
labels = data[:, -1:]
learning_rate = 0.01

weights = np.array([
    [1],
    [1]
])

features = np.append(features, np.ones(shape=(6, 1)), axis=1)


def gradient_descent():
    prediction = 1 / (1 + np.exp(-np.dot(features, weights)))
    return np.dot(features.T, prediction - labels)


def train():
    for i in range(1000000):
        slope = gradient_descent()
        print(slope)
        global weights
        weights = weights - learning_rate * slope
        # if abs(slope[0][0]) < 0.1 and abs(slope[1][0]) < 1:
        #     break


if __name__ == '__main__':
    train()
    print(weights)
