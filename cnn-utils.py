import numpy as np
import tensorflow as tf


def get_weights(shape):
    # Random normal weights
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def get_bias(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def get_conv_layer():
    pass


def get_res_layer():
    pass
