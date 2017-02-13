# download an image nd call it packt.jpeg

# import matplotlib library in bash - sudo apt-get install python-matplotlib
import matplotlib.image as mp_image
filename = "packt.jpeg"
input_image = mp_image.imread(filename)

print('input dim = {}'.format(input_image.ndim))
print('input shape = {}'.format(input_image.shape))

# out put = ( 100, 181, 3) which is 100 pixels high, 144 pixels wide and 3 colours deep

# now we will visualise the image

import matplotlib.pyplot as plt
plt.imshow(input_image)
plt.show()

# we will start tensorflow and create a placeholder to hold the rgb values
import tensorflow as tf
my_image = tf.placeholder("uint8", [None, None, 3])  # we only want to store the 3 rgb values

# we now create a sub-image, a slice of the full image, using the 3 rgb values
slice = tf.slice(my_image, [10,0,0], [20,-1,-1])

# now build a tensorflow session
with tf.Session() as sess:
    result = sess.run(slice, feed_dict={my_image: input_image})
    print(result.shape)

plt.imshow(result)   # Note, it will regenerate 1st image which you need to click to close so the next can be rendered
plt.show()

# now we will transform the image - flip it over and turn 90 degrees
import tensorflow as tf

#associate the input image with a variable x
x = tf.Variable(input_image, name='x')

# initialise our image
model = tf.initialize_all_variables()

# now build a session
with tf.Session() as session:
    x = tf.transpose(x, perm=[1,0,2])
    session.run(model)
    result=session.run(x)

plt.imshow(result)
plt.show()
