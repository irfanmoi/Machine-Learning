import tensorflow as tf
tf.set_random_seed(777)	
# y = mx + b, where m and b are vars, y represents prediction
# cost function is used to optimize weights


'''
HYPOTHESIS FUNCTION: H(x) = Wx + b 
COST FUNCTION: cost(W,b) = (1/m)m(sigma)i=1(H(x^(i) - y^(y))^2 (MSE) 
'''
# (1/m)m(sigma)i=1 is the mean
# m is the 
train_x = [1, 2, 3] 
train_y = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf. Variable(tf.random_normal([1]), name = 'bias')

#
hypo = train_x*W + b

# H(x) = Wx + b 
# W is weight, b is bias

# cost/loss function (mean of (hypothesis - train_y)^2)
# measures the distance between dataset and prediction
cost = tf.reduce_mean(tf.square(hypo - train_y))