#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 00:08:06 2017

@author: firefreezing
"""

#!/usr/bin/env python
#%%
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import timeit

#%%
root_dir = "/Users/firefreezing/DataScience/DeepLearning/DL_test"
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
logdir = "{}/tf_logs/ffn-run-{}/".format(root_dir, now)

#%%
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us

#%%
mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

#%%
print(trX.shape)    # 784 dim with 55000 records
print(trY.shape)    # 10 labels with 55000 records

print(teX.shape)    # 784 dim with 10000 records
print(teY.shape)    # 10 labels with 10000 records

dim_X = trX.shape[1]
dim_Y = trY.shape[1]

dim_h = 625   # dimension of hidden units in the hidden layer

#%%
# Construct the computation graph
X = tf.placeholder("float", [None, dim_X])
Y = tf.placeholder("float", [None, dim_Y])

w_h = init_weights([dim_X, dim_h]) # create symbolic variables
w_o = init_weights([dim_h, dim_Y])

py_x = model(X, w_h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # construct an optimizer
predict_op = tf.argmax(py_x, 1)


cost_summary = tf.summary.scalar('Cross_Entropy', cost)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())


#%%
n_epochs = 10
learning_rate = 0.05
mini_batch_size = 128

#%%
# Launch the graph in a session
with tf.Session() as sess:
    # need to initialize all variables
    tf.global_variables_initializer().run()
    
    step = 0
    
    for epoch in range(n_epochs):
        for start, end in zip(range(0, len(trX), mini_batch_size), 
                              range(mini_batch_size, len(trX)+1, mini_batch_size)):
            
            # report the cost for each iteration of the mini-batch
            summary_str = cost_summary.eval(feed_dict = {X: trX[start:end], 
                                                         Y: trY[start:end]})
            step = step + 1
            # write out to log directory to be used in tensorboard
            file_writer.add_summary(summary_str, step)
            
            # excute GD on each mini-batch
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        
        print(epoch, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX}))) # accuracy on the test set

    file_writer.close()