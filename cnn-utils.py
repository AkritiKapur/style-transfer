import numpy as np
import tensorflow as tf


def get_weights(shape):
    # Random normal weights
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def get_bias(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def get_conv_layer(filter_height, filter_width, n_input_channels, n_channels, stride=1):
    """
    Returns a conv layer (tensor variable)
    :param filter_height: Conv filter height
    :param filter_width: Conv filter width
    :param n_channels: Number of filters in the conv layer (current layer)
    :param n_input_channels: number of channels in the input layer (previous layer)
    :param stride:
    :return: Conv layer (tensor variable)
    """

    shape = [filter_height, filter_width, n_input_channels, n_channels]
    weight = get_weights(shape)


def get_res_layer():
    pass
