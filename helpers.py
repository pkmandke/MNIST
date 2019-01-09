# Helpers for cnn trials

import tensorflow as tf

"""
Fetching data - only once

from tensorflow.examples.tutorials.mnist import input_data

DATA_PATH = '/home/prathamesh/undergrad/btech_project/misc/mnist_data'

mnist_dat = input_data.read_data_sets(DATA_PATH, one_hot=True);
"""

def max_pool(input_, kernel=[1,2,2,1], stride=[1,2,2,1], padd='SAME'):
    return tf.nn.max_pool(input_, ksize=kernel, strides=stride, padding=padd)

def get_weights(size):
    return tf.Variable(tf.random_normal(shape=size, stddev=0.4))

def get_bias(size):
    return tf.Variable(tf.constant(0.1, shape=size))

def full_conn(input_, size):
    W = get_weights([input_.get_shape()[1],size])
    bias = get_bias([size]);
    return tf.matmul(input_, W) + bias

def conv_layer(input_, filtr, stride=[1,1,1,1], padd='SAME'):
    W = get_weights(filtr);
    bias = get_bias([filtr[3]])
    return tf.nn.relu(tf.nn.conv2d(input_, W, strides=stride, padding=padd) + bias)
