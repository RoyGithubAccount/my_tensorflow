"""
Work in progress

"""
#help the program to find the mnist data
import sys
sys.path.append('/tensorflow/lib/python3.5/site-packages/tensorflow/examples/tutorials/mnist')
# set up variables and constants
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
