import numpy as np

data_cars = np.loadtxt('cars.csv', delimiter=',', skiprows=1, usecols=(1, 4, 5))

horsepowers = data_cars[:, 1:2]
weights = data_cars[:, 2:3]
features = np.append(horsepowers, weights, axis=1)
print(features)

mpgs = data_cars[:, 0]

"""
mpg = m * horsepower + b
mse = (1/n)*Σ(guess - actual)**2
mse = (1/n)*Σ(m*horsepower + b - actual)**2

d(mse)/dm = (2/n)*Σ(m*horsepower+b -actual)*horsepower
d(mse)/db = (2/n)*Σ(m*horsepower+b -actual)
"""
m1 = 1
m2 = 1
b = 1
learning_rate = 0.00001


def gradient_descent():
    m1_slope = 0
    for i, feature in enumerate(features):
        m1_slope = m1_slope + (2 / len(features)) * ((m1 * feature[0] + m2 * feature[1] + b) - mpgs[i]) * feature[0]

    m2_slope = 0
    for i, feature in enumerate(features):
        m2_slope = m2_slope + (2 / len(features)) * ((m1 * feature[0] + m2 * feature[1] + b) - mpgs[i]) * feature[1]

    b_slope = 0
    for i, feature in enumerate(features):
        b_slope = b_slope + (2 / len(features)) * ((m1 * feature[0] + m2 * feature[1] + b) - mpgs[i])

    return m1_slope, m2_slope, b_slope


def train():
    for i in range(1000000000):
        global m1, m2, b
        m1_slope, m2_slope, b_slope = gradient_descent()
        print("m1_slope:", m1_slope, "m2_slope:", m2_slope, "b_slope:", b_slope)
        m1 = m1 - learning_rate * m1_slope
        m2 = m2 - learning_rate * m2_slope
        b = b - learning_rate * b_slope
        if abs(m1_slope) < 0.1 and abs(m2_slope) < 0.1 and abs(b_slope) < 0.1:
            break


if __name__ == '__main__':
    train()
    print("y = {}x1+{}x2+{}".format(m1, m2, b))
