import tensorflow as tf
import numpy as np

# build two integer arrays
matrix1 = np.array([(2,2,2), (2,2,2), (2,2,2)], dtype='int32')
matrix2 = np.array([(1,1,1), (1,1,1), (1,1,1)], dtype='int32')

# visualize them
print "matrix1 = "
print matrix1

print "matrix2 = "
print matrix2

print("Next we transform the matricies in to a tensor data structure")
matrix1 = tf.constant(matrix1)
matrix2 = tf.constant(matrix2)

print("now converted the tensors are ready to be manipulated with operators, here the matrix sum is found")
matrix_product = tf.matmul(matrix1, matrix2)
matrix_sum = tf.add(matrix1, matrix2)

print("find the determinant")
matrix_3 = np.array([(2,7,2), (1,4,2), (9,0,2)], dtype='float32')
print "matrix 3 ="
print matrix_3

matrix_det = tf.matrix_determinant(matrix_3)

print("Now we create a graph and run the session")
with tf.Session() as sess:
    result1 = sess.run(matrix_product)
    result2 = sess.run(matrix_sum)
    result3 = sess.run(matrix_det)

# print out results
print "matrix1 * matrix2 ="
print result1

print "matrix1 + matrix2 ="
print result2

print "[(2,7,2), (1,4,2), (9,0,2)] = matrix3 and its determinant result ="
print result3