import numpy as np
tensor_2d = np.array([(1,2,3,4),(4,5,6,7),(8,9,10,11),(12,13,14,15)])
print tensor_2d
print tensor_2d[3][3]
print tensor_2d[0:2, 0:2]
# view rank of tensor
print("Rank of tensor is  ")
print tensor_2d.ndim
print("Shape of tensor, also a tuple, is  ")
print tensor_2d.shape
print("Data type of tensor is  ")
print tensor_2d.dtype
print("Now we convert the python tuple in to a TensorFlow tensor ")
print("We start by importing TF")
import tensorflow as tf
print("We now convert")
tf_tensor=tf.convert_to_tensor(tensor_2d, dtype=tf.float64)
print("We now run a session")
with tf.Session() as sess:
    print("Stats out of TensorFlow calling tf_tensor")
    print sess.run(tf_tensor)
    # next line adapted from book version so it works. book had tensor_2d[3][3]
    print sess.run(tf_tensor[3,3])
    print sess.run(tf_tensor[0:2, 0:2])