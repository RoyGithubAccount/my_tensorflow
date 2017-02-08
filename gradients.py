import tensorflow as tf

# create independent variable
x = tf.placeholder(tf.float32)

# build function we want to find the derivative of and with a value assigned to x
y = 5*x*x*x

# call tf.gradients()
var_grad = tf.gradients(y,x)

# to evaluate we need a session
with tf.Session() as session:
    var_grad_val = session.run(var_grad, feed_dict={x:2}) # feed_dict is the value we want to plug in for x
    print " The value of dy/dx for the entered formula and a value of x ="
    print(var_grad_val)