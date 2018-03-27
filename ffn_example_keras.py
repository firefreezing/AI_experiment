#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 07:07:20 2018

@author: firefreezing
"""

#%%
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.callbacks import TensorBoard

from datetime import datetime

#%%
root_dir = "/Users/firefreezing/DataScience/DeepLearning/DL_test"
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
logdir = "{}/tf_logs/ffn-keras-run-{}/".format(root_dir, now)

#%%
from keras.datasets import mnist
# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#%%
print(X_train.shape)

#%% 
from matplotlib import pyplot as plt

plt.imshow(X_train[1])

#%%
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

#%%

model = Sequential()

# model.add(Flatten())
model.add(Dense(625, input_dim = 784, activation = 'sigmoid')) # if using relu, the performance will significantly better after 10 epochs 
model.add(Dense(10, activation = 'softmax'))

#%%
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'sgd',
              metrics = ['accuracy'])

#%%
# callback TensorBoard
tbCallBack = TensorBoard(log_dir=logdir, histogram_freq=0, 
                         write_graph=True, write_images=True)

#%%
model.fit(X_train, Y_train,
          batch_size = 128,
          nb_epoch = 10,
          verbose = 1,
          callbacks = [tbCallBack]  # callback TensorBoard
          )

#%%
score = model.evaluate(X_test, Y_test, verbose = 0)
print(score)