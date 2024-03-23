import numpy as np

data_cars = np.loadtxt('cars.csv', delimiter=',', skiprows=1, usecols=(1, 4, 5))

np.random.shuffle(data_cars)

test_data = data_cars[:40]
train_data = data_cars[40:]

np.savetxt('train.csv', train_data, delimiter=',', fmt='%f')
np.savetxt('test.csv', test_data, delimiter=',', fmt='%f')