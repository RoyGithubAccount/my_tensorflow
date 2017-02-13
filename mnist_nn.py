"""
Neural network that predicts what a hand written number of out of MNIST data set

The objective of this program is to understand how a NN is wired together and how it works

"""

#help the program to find the mnist data
import sys
sys.path.append('/tensorflow/lib/python3.5/site-packages/tensorflow/examples/tutorials/mnist')

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# set up variables and constants
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# define our model
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# define number of classes, here for 0 to 9 it ios 10 classes
n_classes = 10

# chunk data up so we can work in the limits of RAM available to us
batch_size = 100

# define some placeholders we pass values through - x is the data and y are the labels
x = tf.placeholder('float', [None, 784])
# the list is the shape of the tensor - 0 Height and 784 wide (28 pixels x 28 pixels)

y = tf.placeholder('float')

# define the NN
def neural_network_model(data):

    # how data will flow through the NN
    hidden_1_layer = {'weight': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    # our model for each layer is (input_data * weight) + biases

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['biases'])

    #we will use rectified linear function, relu, as the activation and threshold function
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weight']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weight']) + output_layer['biases']

    return output

# now we want to run data through the NN model we defined above
def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    # we want to minimise cost so...
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # set the number of epochs - cycles through 'feed forward' and 'back propagation' - how many epochs?
    hm_epochs = 10
    # start a session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss +=c
                print("Epoch ", epoch, "Completed out of ", hm_epochs, "Loss: ", epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print("Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_neural_network(x)



