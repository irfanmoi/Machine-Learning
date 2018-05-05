import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to fix some random warning that kept coming up 

hello = tf.constant('Hello World and TensorFlow!')

# start a tensorflow session
sess = tf.Session()

# running the operation

print(sess.run(hello)) # 'b'is Bytes literals

'''
TENSOR FLOW MECHANICS:
1) build graph using TF operations 
2) feed data and run operation (graph) SYNTAX :sess.run(op)
3) update vars in the graph + return values
'''

# Build graph (1)

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # float32 is implied
node3 = tf.add(node1, node2) # adds 4 + 3

# Run graph (2)

print("node 1 and node 2 session:", sess.run([node1, node2]))
print("node 3 session:", sess.run(node3))

# Placeholder: a var which will be assigned data at a later date.
# In tensorflow, data is fed data into the graph through these placeholders.

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node_op = a + b # OR tf.add(a, b)

# starts session
# SYNTAX: sess.run(op, var input)

feed_dict_input = {a: 3, b: 4.5}
print(sess.run(adder_node_op, feed_dict_input)) 
