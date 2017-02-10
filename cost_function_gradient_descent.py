# uses tensorflow for Linear Regression where constants A and B are unknown
# introduces cost functions and gradient descent
# aim is to predict y as a function of x in y = Ax + b

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

# Define A and b as tf variables
A = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

# initialise x_point and y_point as in linear_regression.py
x_point = []
y_point = []

# generate 300  to store in x_point and y_point
for i in range(300):
    x_value = np.random.normal(0.0, 0.3)
    y_value = np.random.normal(0.0, 0.05)
    #x_value = i**3
    #y_value = i**2
    x_point.append([x_value])
    y_point.append([y_value])

# bind y  to x in a linear relationship
y = A * x_point + b

# define the Cost Function
# has parameters containing a pair of values for A and b
# returns a value that estimates how correct the predictions for A and b are using mean square error
cost_function = tf.reduce_mean(tf.square(y_point - y))

"""
xxxxxxxxxxxxxxxx  Gradient Descent Optimizer  xxxxxxxxxxxxxxxxxxxx
We will optimise the cost_function with gradient descent
1. we pick any value
2. we differentiate it to find the slope of the line (call it "dy/dx #1")
3. we pick a second number
4. we differentiate that (call it "dy/dx #2")
5. we compare dy/dx #1 and #2 to find which way gets us to a lower number
6. we keep repeating this until we hit the minima after which the direction will start to rise
7. beware false minima !!!
"""

optimizer = tf.train.GradientDescentOptimizer(0.5)
# 0.5 is the learning rate, if it's too big we jump over the optimal point, if it's too small
# it takes a long time to get to the solution
# we would tune this rate based on experience of it working

# we define 'train' as the result of the application of the cost_function(optimizer) through
# its 'minimize function
train = optimizer.minimize(cost_function)

"""
xxxxxxxxxxxxxxxx Testing the model  xxxxxxxxxxxxxxxx
"""
model = tf.initialize_all_variables()

# we will use 20 steps to find the optimimum
with tf.Session() as session:
    # we perform the simulation on our model
    session.run(model)
    for step in range(0, 21):
        # for each iteration we execute the optimization step
        session.run(train)
        # every 5 steps we print a pattern of dots
        if (step % 5) == 0:
            plt.plot(x_point, y_point, 'o',
                     label='step = {}'.format(step))
            # straight line come from below
            plt.plot(x_point,
                     session.run(A) *
                     x_point +
                     session.run(b))
            plt.legend()
            plt.show(block=False)
            time.sleep(1)
            plt.close()