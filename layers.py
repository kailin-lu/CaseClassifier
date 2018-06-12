import tensorflow as tf
tf.set_random_seed(10)

def conv_layer(input, kernel, channels_in, channels_out, index): 
    with tf.name_scope('conv{}'.format(index)):
        w = tf.get_variable(shape=[kernel, input.shape[2], channels_in, channels_out], 
                            initializer=tf.contrib.layers.xavier_initializer(), 
                            name='conv-weight-{}'.format(index))
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]))
        conv = tf.nn.conv2d(input, w, 
                            strides=[1,1,1,1], 
                            padding='VALID', 
                            data_format='NHWC', 
                            name='conv{}'.format(index))
        activation = tf.nn.relu(conv + b)
    return activation