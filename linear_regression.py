import numpy as np

data = np.array([
    [80, 200],
    [95, 230],
    [104, 245],
    [112, 274],
    [125, 259],
    [135, 262]
])
m = 1
b = 1
featuref = data[:, 0]
y_array = data[:, -1]

learning_rate = 0.00001


def gradient_descent():
    b_slope = 0
    for index, x in enumerate(feature):
        b_slope = b_slope + m*x+b-y_array[index]
        # b_slope = b_slope*2 / len(x_array)
    # print("mse对b求导={}".format(b_slope))

    m_slope = 0
    for index, x in enumerate(feature):
        m_slope = m_slope + (m*x+b - y_array[index])*x
        # m_slope = m_slope * 2 / len(x_array)
    # print("mse对m求导={}".format(m_slope))
    return m_slope, b_slope


def train():
    for i in range(1, 10000000):
        m_slope, b_slope = gradient_descent()
        global m
        m = m - learning_rate * m_slope
        global b
        b = b - learning_rate * b_slope
        if abs(m_slope) < 0.1 and abs(b_slope) < 0.1:
            break


if __name__ == '__main__':
    train()
    print("m={},b={}".format(m, b))

