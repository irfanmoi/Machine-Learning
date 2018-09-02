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

'''

Logistic Regression is a classifier. 
In other words, it predicts the probability of categorization, according to the
input. Uses logistic sigmoid function to return probability value, which can be 
mapped to 2 or more discrete categories.

Types of logistic regression

1. binary (0 or 1) 
2. multi (blue, red, or yellow)
3. ordinal (low, medium, high)

activation: softmax function
cost/loss: cross-entropy/log loss function
optimizer: stochastic gradient descent 

ABOUT:

-softmax function: 
taking exponential of each element and normalizing each vector.

-normalized vector:
dividing vector by magnitude and reduce to unit vector,
gets rid of scaling information, in other words, 
magnitude of normalized vector is 1.

-one-hot encoding:
symbolize a number using a vector with 10 elements,
which are 0 and 1s.
for example, a "4" in one-hot would look like:
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

                    
BACKGROUND:
- every image in MNIST is 28*28 image
- in other words, there are 784 pixels
- the pixels are used for input

Y = softmax(Wx + b)

Y: hypothesis, Y[100, 10]
W: weight, W[784, 10]
x: images, x[100, 784]
b: bias, b[1, 10]


In this model, variables are the weights (W) and
bias (b), which are determined by the activation.

The placeholders x and desired_outcome are
filled during training phase. The variable X holds
the input images, and the desired_outcome holds
the labels which accompany the images.

'''
# MNIST: large database of handwritten digits, 60k for training, 10k for testing

# "train", "test", "validation" can be accessed,
# inside each, "images", "labels", "num_examples" can be accessed

print(MNIST.train.num_examples, MNIST.test.num_examples, MNIST.validation.num_examples)
# images =  total number of images*pixels(784)
# labels = total number of labels*categories(0-9)
print(MNIST.train.images.shape, MNIST.train.labels.shape)
print(np.max(MNIST.train.images), np.min(MNIST.train.images))

# visual
plot.imshow(np.reshape(MNIST.train.images[64, :], (28, 28)), cmap="gray")
#plot.show()

# linearize each image
x = tf.placeholder(tf.float32, [None, 784])
# labels
desired_outcome = tf.placeholder(tf.float32, [None, 10])

# a non-empty tensor
# all values set to 0
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# vector of non-normalized hypothesis
logits = tf.matmul(x, W) + b
# activation
y = tf.nn.softmax(logits)
# cost/loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=desired_outcome, logits=logits))

# TRAINING STEP
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cross_entropy)

session = tf.Session()
session.run(tf.global_variables_initializer())

for i in range(9999):
    batch_0, batch_1 = MNIST.train.next_batch(128)
    session.run(optimizer, feed_dict={x: batch_0, desired_outcome: batch_1})

# argmax: largest value across axis
prediction_accuracy = tf.equal(tf.argmax(logits, axis=1), tf.argmax(desired_outcome, axis=1))
precision = tf.reduce_mean(tf.cast(prediction_accuracy, dtype=tf.float32))
print(session.run(precision, feed_dict={x: MNIST.test.images, desired_outcome: MNIST.test.labels}))
