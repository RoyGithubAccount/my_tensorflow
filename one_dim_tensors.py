import numpy as np
tensor_1d = np.array([1.3, 1, 4.0, 23.99])
print tensor_1d
print tensor_1d[0]
print tensor_1d[2]
# view rank of tensor
print("Rank of tensor is  ")
print tensor_1d.ndim
print("Shape of tensor, also a tuple, is  ")
print tensor_1d.shape
print("Data type of tensor is  ")
print tensor_1d.dtype
print("Now we convert the python tuple in to a TensorFlow tensor ")
print("We start by importing TF")
import tensorflow as tf
print("We now convert")
tf_tensor=tf.convert_to_tensor(tensor_1d, dtype=tf.float64)
print("We now run a session")
with tf.Session() as sess:
    print("Stats out of TensorFlow calling tf_tensor")
    print sess.run(tf_tensor)
    print sess.run(tf_tensor[0])
    print sess.run(tf_tensor[2])