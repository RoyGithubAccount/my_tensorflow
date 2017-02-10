# not really a linear regression as it just put random
# points around a given straight line equation
# page 68 in Zaccone's book - next prog we use cost functions and gradient
# descent to do better linear regression

import numpy as np

# number of points we want to draw
number_of_points = 500

# initialise 2 lists
x_point = []
y_point = []

# set 2 constants we'll plug in to y = 0.22x + 0.78
a = 0.22
b = 0.78

# generate 300 random points around y = 0.22x + 0.78
for i in range(number_of_points):
    x = np.random.normal(0.0, 0.5)
    y = a*x + b +np.random.normal(0.0, 0.1)
    x_point.append([x])
    y_point.append([y])


# view in matplotlib
import matplotlib.pyplot as plt
plt.plot(x_point, y_point, 'o', label='Input Data')
plt.legend()
plt.show()

