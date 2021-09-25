import tensorflow as tf

input = tf.Variable(tf.random_normal([100, 28, 28, 1]))
filter = tf.Variable(tf.random_normal([5, 5, 1, 6]))

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.initialize_all_variables())

op = tf.nn.conv2d(input, filter, strides = [1, 1, 1, 1], padding = 'VALID')
out = sess.run(op)

# TODO install conda, and set a new enviroment with conda to install cuda 10.0 coreesponding to tf1.14.0
# TODO conda installed, but environment not set properly, try when have time