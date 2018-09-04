import tensorflow as tf 
import numpy as np
# fix some error
ld_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plot  
# import MNIST examples
# Functions for downloading and reading MNIST data
import tensorflow.examples.tutorials.mnist.input_data as input_data
MNIST = input_data.read_data_sets("MNIST", one_hot=True)


# linearize each image
x = tf.placeholder(tf.float32, [None, 784])
# labels
y_ = tf.placeholder(tf.float32, [None, 10])

learning_rate = tf.placeholder(tf.float32)

p_neuron = tf.placeholder(tf.float32)

i = tf.placerholder(tf.int32)

W_1 = tf.Variable(tf.truncated_normal([784, 200], stddev=0.1))
b_1 = tf.Variable(tf.ones([200]/10))

W_2 = tf.Variable(tf.truncated_normal([200, 100], stddev=0.1))
b_2 = tf.Variable(tf.ones([100]/10))

W_3 = tf.Variable(tf.truncated_normal([100, 60], stddev=0.1))
b_3 = tf.Variable(tf.ones([60]/10))

W_4 = tf.Variable(tf.truncated_normal([60, 30], stddev=0.1))
b_4 = tf.Variable(tf.ones([30]/10))

W_5 = tf.Variable(tf.truncated_normal([30, 10], stddev=0.1))
b_5 = tf.Variable(tf.zeros([10]))

y_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)
y_2 = tf.nn.relu(tf.matmul(y_1, W_2) + b_2)
y_3 = tf.nn.relu(tf.matmul(y_2, W_3) + b_3)
y_4 = tf.nn.relu(tf.matmul(y_3, W_4) + b_4)
logits = tf.matmul(y_4, W_5) + b_5
y = tf.nn.softmax(logits)

cost = 100*(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_)))

learning_rate = 0.0001 + tf.train.exponential_decay(0.0003, i, 2000)


