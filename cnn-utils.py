import numpy as np
import tensorflow as tf


def get_weights(shape):
    # Random normal weights
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def get_bias(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def get_conv_layer(input, filter_height, filter_width, n_input_channels, n_channels, stride=[1, 1, 1, 1]):
    """
    Creates a conv layer (tensor variable)
    :param input: Input from the previous layer
    :param filter_height: Conv filter height
    :param filter_width: Conv filter width
    :param n_channels: Number of filters in the conv layer (current layer)
    :param n_input_channels: number of channels in the input layer (previous layer)
    :param stride:
    :return: Conv layer (Tensor) and filter (Tensor)
    """
    shape = [filter_height, filter_width, n_input_channels, n_channels]

    # Add filters (weights)
    weight = get_weights(shape)

    # Add bias for each filter
    bias = get_bias(n_channels)

    # Add convolution layer
    # TODO: check padding..
    layer = tf.nn.conv2d(input=input, filter=weight, strides=stride, padding='SAME')

    layer += bias

    # Add a relu unit
    layer = tf.nn.relu(layer)

    return layer, weight


def get_res_layer():
    """
    Creates a residual layer
    :return:
    """
    pass
