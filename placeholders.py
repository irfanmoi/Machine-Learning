import tensorflow as tf
import numpy as np
tf.set_random_seed(777)
'''

A placeholder basically a variable which is assigned data at a later date.
In other words, it enables us to produce operations, and construct a graph
and make computations.

In tensorflow, data is fed to the graph through placeholders.

Constants are initialized when they are defined.
However, to initialize variables, tf.global_variables_iniatializer()
needs to be called.


config=tf.ConfigProto(log_device_placement=True) - makes sure you log GPU/CPU device that is assigned to operation
config=tf.ConfigProto(allow_soft_placement=True) - use soft constraints for the device placement 

'''

a = tf.placeholder("float", None)
b = a*a

with tf.Session() as session:
    output = session.run(b, feed_dict={a: [1, 2, 3]})
    print("example 1: ", output)

# returns [1. 4. 9.]



    










