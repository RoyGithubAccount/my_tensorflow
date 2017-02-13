import tensorflow as tf
import matplotlib.pyplot as plt

# the uniform variable is a 1 dimensional tensor, with 100 elements,
# containing values between 0 and 1, each with the same probability
# random_uniform(shape, minval, maxval, dtype, seed, name)
uniform = tf.random_uniform([100],minval=0, maxval=1, dtype=tf.float32)  # not using seed or name

#define our session
sess = tf.Session()

# evaluate session using eval()
with tf.Session() as session:
    print(uniform.eval())
    plt.hist(uniform.eval(), normed=True)
    plt.show()
