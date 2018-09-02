import tensorflow as tf 
import numpy as np
ld_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
import matplotlib.pyplot as plot  
# import MNIST examples
# Functions for downloading and reading MNIST data
import tensorflow.examples.tutorials.mnist.input_data as input_data
# one_hot means a sparse vector for every observation
# every observation = class label = 1, every other = 0
# 
MNIST = input_data.read_data_sets("MNIST_data/", one_hot=True)

'''

Logistic Regression is a classifier. 
In other words, it predicts the probability of categorization, according to the
input. Uses logistic sigmoid function to return probability value, which can be 
mapped to 2 or more discrete categories.

Types of logistic regression

1. binary (0 or 1) 
2. multi (blue, red, or yellow)
3. ordinal (low, medium, high)

cost function: cross-entropy/log loss function

'''
# MNIST: large database of handwritten digits, 60k for training, 10k for testing

# "train", "test", "validation" can be accessed,
# inside each, "images", "labels", "num_examples" can be accessed

print(MNIST.train.num_examples, MNIST.test.num_examples, MNIST.validation.num_examples)

# the images are n-dimension array (n_observations*n_labels)
# labels are (n_observations*n_labels)
# every observation has a class label = 1 or 0

print(MNIST.train.images.shape, MNIST.train.labels.shape)

print(np.max(MNIST.train.images), np.min(MNIST.train.images))

plot.imshow(np.reshape(MNIST.train.images[64, :], (28, 28)), cmap="gray")
#plot.show()

#linearize each image

x = tf.placeholder(tf.float32, [None, 784]) 
desired_outcome = tf.placeholder(tf.float32, [None, 10])
# a non-empty tensor

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
logits = tf.matmul(x, W) + b 
y = tf.nn.softmax(logits)

# cost/loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=desired_outcome, logits=logits))

# TRAINING STEP
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(cross_entropy)

# argax: largest value across axis
# in this case, 1, because of one_hot=1

session = tf.Session()
session.run(tf.global_variables_initializer())

for i in range(9999):
    batch_0, batch_1 = MNIST.train.next_batch(128)
    session.run(optimizer, feed_dict={x: batch_0, desired_outcome: batch_1})


prediction_accuracy = tf.equal(tf.argmax(logits, axis=1), tf.argmax(desired_outcome, axis=1))
precision = tf.reduce_mean(tf.cast(prediction_accuracy, dtype=tf.float32))
print(session.run(precision, feed_dict={x: MNIST.test.images, desired_outcome: MNIST.test.labels}))



