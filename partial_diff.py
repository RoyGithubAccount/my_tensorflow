"""xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

Partial differentiation equation (PDE) - partial derivitives of an unknown
function of several independent variables

Here we model the 2D surface of a pond to show waves
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# functions needed below
def make_kernal(a):
    a = np.asarray(a)
    a = a.reshape(list(a.shape) + [1,1])
    return tf.constant(a, dtype=1)

def simple_conv(x, k):
    x = tf.expand_dims(tf.expand_dims(x, 0), -1)
    y = tf.nn.depthwise_conv2d(x, k, [1,1,1,1], padding='SAME')
    return y[0, :, :, 0]

def laplace(x):
    laplace_k = make_kernal([[0.5, 1.0, 0.5],
                             [1.0, -6.0, 1.0],
                             [0.5, 1.0, 0.5]])
    return simple_conv(x, laplace_k)

# define pond as 150 x 150
N = 150

# initial condition is pond at t = 0
u_init = np.zeros([N, N], dtype=np.float32)
ut_init = np.zeros([N, N], dtype=np.float32)

# 40 random rain drops to hit the pond
for n in range(40):
    a,b = np.random.randint(0, N, 2)   # returns integers from 0 to N on 2D shape
    u_init[a,b] = np.random.uniform()

# Model building

# define fundamental parameters as tf placeholders and a time step of the simulation
eps = tf.placeholder(tf.float32, shape=())

# we need to define a damping coefficient
damping = tf.placeholder(tf.float32, shape=())

# redefine our starting tensors in tf as they will change over time
U = tf.Variable(u_init)
Ut = tf.Variable(ut_init)

# now we can build our PDE model
U_ = U + eps * Ut
Ut_ = Ut + eps * (laplace(U) - damping * Ut) # using Laplace transform to resolve PDE

# using tf 'group' operator define pond as it evolves in time t
# group operator gathers multiple operations as a single operation
step = tf.group(U.assign(U_), Ut.assign(Ut_))

# Graph Execution

i = 0
# initialise tf variables
with tf.Session() as session:
    tf.initialize_all_variables().run()

    # run simulation
    #for i in range(1000):
        #step.run({eps: 0.03, damping: 0.04})
        #if i % 50 == 0:
            #clear_output   # function unknown
            #plt.imshow(U.eval())
            #plt.show() # needs to manually close plot before next cycle through loop starts

    while (i <= 1000):
        step.run({eps: 0.03, damping: 0.04})
        if i % 50 == 0:
            os.system('clear')
            plt.imshow(U.eval())
            plt.show(block=False)
            time.sleep(1)
            plt.close()
        i = i + 1
