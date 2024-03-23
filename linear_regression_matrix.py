import numpy as np

data_cars = np.loadtxt('cars.csv', delimiter=',', skiprows=1, usecols=(1, 4, 5))
feature = data_cars[:, 1:3]
mpgs = data_cars[:, 0:1]

m1 = 1
m2 = 1
b = 1
weight = np.array([
    [m1],
    [m2],
    [b]
])
learning_rate = 0.00001
def gradient_descent():
    feature_add_one = np.append(feature, np.ones(shape=(len(feature), 1)), axis=1)
    slope = np.dot(feature_add_one.T, np.dot(feature_add_one, weight) - mpgs)/len(feature_add_one)*2
    return slope


def train():
    for i in range(1000000000):
        slope = gradient_descent()
        global weight
        weight = weight - slope * learning_rate
        print(slope)
        if abs(slope[0][0]) < 1 and abs(slope[1][0]) < 1 and abs(slope[2][0]) < 1:
            break

if __name__ == '__main__':
    train()
    print('mpg = {}horsepower +{}weight + {}'.format(weight[0][0], weight[1][0], weight[2][0]))
