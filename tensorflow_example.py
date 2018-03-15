# -*- coding: utf-8 -*-
"""
This is a quick example of running simple linear regression (SLR) using Tensorflow

author: firefreezing 
date: 03/11/2018

reference: chapter 9 from Hands-On Machine Learning with Scikit-Learn and TensorFlow, by Aurelien Geron
"""

#%%
import tensorflow as tf

#%% a quick tf example
# create the computation graph
x = tf.Variable(3, name = 'x')
y = tf.Variable(2, name = 'y')
f = x*y + 2

# run the graph
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
    
print(result)

#%%
str(x)

x.graph is tf.get_default_graph()

#%% linear regression with TensorFlow

# reset the default graph
tf.reset_default_graph()

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler 

# loader for the California housing data 
help(fetch_california_housing)

housing = fetch_california_housing()   # load the data
print(str(housing))

m, n = housing.data.shape

housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data] # what .data does here?
# c_ = bind_c

# why .data not showing in the auto-complete of the "housing." - try the same thing in the ipython console

#%% Method 1 - solve the SLR problem using regression formula: theta = (X^t * X)^{-1} * X^t * y

# the computation graph
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(X)

theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)  # the equation for SLR

with tf.Session() as sess:
    theta_value = theta.eval()
    print('coefficient estimate from formula:\n', theta_value)
    

#%% Method 2 - solve the SLR problem using full strength of TF

# the computation graph
n_epochs = 1000
learning_rate = 0.01

X_full, y_full = housing.data, housing.target

scaled_X_full = StandardScaler().fit_transform(X_full)
scaled_y_full = StandardScaler().fit_transform(y_full.reshape(-1,1))

scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_X_full]

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(scaled_y_full, dtype=tf.float32, name="Y")

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")

y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y

mse = tf.reduce_mean(tf.square(error), name="mse")

# two ways to calculate the gradients
# 1. through calculus - accurate but not possible for all mse forms
# gradients = 2/m * tf.matmul(tf.transpose(X), error)  # the fomula is calculated by taking the derivative of the MSE

# 2. using autodiff - TF way to automatically and efficiently calculate the gradient
# gradients = tf.gradients(mse, [theta])[0]
# training_op = tf.assign(theta, theta - learning_rate * gradients) # the assign() function creates a node that will assign a new value to a variable

# 3. using an optimizer to make the code even faster: 
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
## other optimizer choices
#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

training_op = optimizer.minimize(mse)
    
    
    
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    
    best_theta = theta.eval()
 
print('coefficient estimate from GD with TF implementation:\n', best_theta) 


#%% Method 3 - solve the SLR problem using gradient descent

# # reset the default graph
# tf.reset_default_graph()

# the computation graph
n_epochs = 1000
learning_rate = 0.01

X_full, y_full = housing.data, housing.target

scaled_X_full = StandardScaler().fit_transform(X_full)
scaled_y_full = StandardScaler().fit_transform(y_full.reshape(-1,1))

scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_X_full]

# make X and y placeholder nodes
X = tf.placeholder(tf.float32, shape=(None, n+1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

batch_size = 100
n_batches = int(np.ceil(m / batch_size))

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")

y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y

mse = tf.reduce_mean(tf.square(error), name="mse")

# using an optimizer to make the code even faster: 
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
## other optimizer choices
#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

training_op = optimizer.minimize(mse)
    
# create a fetch function to fetch the mini-batch
#def fetch_batch(epoch, batch_index, batch_size): * how to incorporate epoch in an efficient way?
def fetch_batch(batch_index, batch_size):
    
    idx_start = batch_index * batch_size
    idx_end = (batch_index + 1) * batch_size - 1
    
    X_batch = scaled_housing_data_plus_bias[idx_start:idx_end, ]
    y_batch = scaled_y_full[idx_start:idx_end]
    
    return X_batch, y_batch
    
    
init = tf.global_variables_initializer()
saver = tf.train.Saver()  # create a saver node at the end of the construction phase

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(batch_index, batch_size)  # add "epoch" when code being updated
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            
        if epoch % 100 == 0:
            print("Epoch", epoch)
            print("MSE = %s" % sess.run(mse, feed_dict = {X: X_batch, y: y_batch}))
            # save_path = saver.save(sess, "/tmp/my_slr.ckpt")
            save_path = saver.save(sess, "/Users/firefreezing/DataScience/DeepLearning/DL_test/weights/my_slr.ckpt")
       
    best_theta = theta.eval()
    # save_path = saver.save(sess, "/tmp/my_slr_final.ckpt")
    save_path = saver.save(sess, "/Users/firefreezing/DataScience/DeepLearning/DL_test/weights/my_slr_final.ckpt")

print('coefficient estimate from GD with TF implementation:\n', best_theta)   

#%% read the graph and weights generated from method 3, and directly computate the results

init = tf.global_variables_initializer()
saver = tf.train.import_meta_graph("/Users/firefreezing/DataScience/DeepLearning/DL_test/weights/my_slr_final.ckpt.meta")  # import the graph's state (i.e. variable values)

with tf.Session() as sess:
    saver.restore(sess, "/Users/firefreezing/DataScience/DeepLearning/DL_test/weights/my_slr_final.ckpt")

print('coefficient estimate from GD with TF implementation:\n', best_theta)   



  
     
    