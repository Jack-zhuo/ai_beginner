import matplotlib.pyplot as plt
import numpy as np
import math

points = np.array([
    [0, 0],
    [0, 5],
    [3, 5],
    [3, 4],
    [1, 4],
    [1, 3],
    [2, 3],
    [2, 2],
    [1, 2],
    [1, 0]
])
plt.plot(points[:, 0], points[:, 1])

# shift two to the right
matrix = np.array([2,0])
new_points = points + matrix
plt.plot(new_points[:, 0], new_points[:, 1])

# rotate 90 degrees counterclockwise
matrix = np.array([
    [math.cos(math.pi/2), -math.sin(math.pi/2)],
    [math.sin(math.pi/2), math.cos(math.pi/2)]
])
rotate_points = np.dot(points, matrix.T)
plt.plot(rotate_points[:, 0], rotate_points[:, 1])

# scale
matrix_scale = np.array([
    [1,1.3],
    [0,1]
])
scale_new_points = np.dot(points, matrix_scale.T)
plt.plot(scale_new_points[:, 0], scale_new_points[:, 1])

plt.xlim(-10, 10)
plt.ylim(-10, 10)

plt.show()
