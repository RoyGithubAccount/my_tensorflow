""" This script is meant to start up TensorFlow automatically
It assumes the following command has been run previously
   $ virtualenv --system-site-packages ~/tensorflow
   To run tf on CPUs
   $ pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl


"""
# start the virtual environment in bash
#  $ source ~/tensorflow/bin/activate
# start python manually and call this file

# hello world
import tensorflow as tf
hello = tf.constant("Hello World")
sess=tf.Session()
print(sess.run(hello))
print("TensorFlow is now loaded")
