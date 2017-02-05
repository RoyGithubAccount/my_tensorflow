import tensorflow as tf

# set constants and variables
a = tf.constant(10, name="a")
b = tf.constant(90, name="b")
y = tf.Variable(a+b*2, name="y")

model = tf.initialize_all_variables()
with tf.Session() as session:
    writer = tf.train.SummaryWriter("./tmp/tensorflowlogs", session.graph_def)
    session.run(model)
    print(session.run(y))
    # close the session so the file is created on disc
    session.close()

"""xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

I could not get the output to show in TensorBoard and after reading online
suspect there is a bug in the pip install method for installing tf; there is a 
missing svg path which the browser will need to render a graph

xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"""
