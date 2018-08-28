import tensorflow as tf
tf.set_random_seed(777)	
# y = mx + b, where m and b are vars, y represents prediction
# cost function is used to optimize weights


'''
HYPOTHESIS FUNCTION: H(x) = Wx + b 
COST FUNCTION: cost(W,b) = (1/m)m(sigma)i=1(H(x^(i) - y^(y))^2 (MSE) 
GRADIENT FUNCTION: cost'(W, b)

1. build graph
2. feed data + run graph
3. update vars in graph + return values
'''

# (1/m)m(sigma)i=1 is the mean

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf. Variable(tf.random_normal([1]), name = 'bias')

train_x = [1, 2, 3]
train_y = [10, 20, 30]

# H(x) = Wx + b 
# W is weight, b is bias
hypo = train_x*W + b

# cost/loss function (mean of (hypothesis - train_y)^2)
# measures the distance between dataset and prediction
cost = tf.reduce_mean(tf.square(hypo - train_y))

# Gradient Descent, to minimize

# iterate through data points,
# (weight + bias vals)
# size of update is controlled
# by learning rate

opt = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = opt.minimize(cost)

# start graph session

# training: iteratively improving prediction
# by looping through data
# update W and b values
# according to gradient of cost 
# function (cost'(W, b))

# hyperparamters: learning rate + # of iterations

session = tf.Session()

# initialize

session.run(tf.global_variables_initializer())

# fit line

for i in range(10000):

    session.run(train)

    if i % 50 == 0:
        print(i, " cost:", session.run(cost), " weight:", session.run(W), " bias:", session.run(b))
    


'''
CONCLUSION:

tf.placeholder is used to feed actual training examples
tf.Variable is trained (changed) from the actual training

'''