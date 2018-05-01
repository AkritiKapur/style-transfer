import numpy as np
import tensorflow as tf


def get_weights(shape):
    # Random normal weights
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def get_bias(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def get_conv_layer(input, filter_height, filter_width, n_input_channels, n_channels, stride=[1, 1, 1, 1], relu=True):
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
    # tf.verify_tensor_all_finite(layer, msg="layer has an issue is it?")

    layer += bias

    layer = batch_normalization(layer, n_channels)

    # Add a relu unit
    if relu:
        layer = add_relu(layer)

    return layer, weight


def get_deconv_layer(input, filter_height, filter_width, n_output_channels, n_input_channels, stride=[1, 1, 1, 1], relu=True):
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
    shape = [filter_height, filter_width, n_output_channels[3], n_input_channels]

    # Add filters (weights)
    weight = get_weights(shape)

    # Add bias for each filter
    bias = get_bias(n_output_channels[3])

    # Add convolution layer
    # TODO: check padding..
    layer = tf.nn.conv2d_transpose(value=input, filter=weight, output_shape=n_output_channels,
                                   strides=stride, padding='SAME')

    layer += bias

    layer = batch_normalization(layer, n_output_channels[3])

    # Add a relu unit
    if relu:
        layer = add_relu(layer)

    return layer, weight


def get_fc_layer(input, n_input, n_output):

    # Set weight
    weight = get_weights([n_input, n_output])

    # Set bias
    bias = get_bias(n_output)

    layer = tf.matmul(input, weight)

    # Add bias
    layer += bias

    layer = add_relu(layer)

    return layer


def add_relu(layer):
    return tf.nn.relu(layer)


def get_res_layer(conv, filter_width, filter_height, n_filters, n_input_channels):
    """
    Creates a residual (tensor variable)
    :param input: Input from the previous layer
    :param filter_height: Conv filter height
    :param filter_width: Conv filter width
    :param n_channels: Number of filters in the conv layer (current layer)
    :param n_input_channels: number of channels in the input layer (previous layer)
    :param stride:
    :return: Conv layer (Tensor) and filter (Tensor)
    """
    temp, _ = get_conv_layer(input=conv, filter_width=filter_width,
                             filter_height=filter_height, n_channels=n_filters, n_input_channels=n_input_channels)

    layer, weight = get_conv_layer(input=temp, filter_width=filter_width,
                   filter_height=filter_height, n_channels=n_filters,
                   n_input_channels=n_input_channels, relu=False)

    layer = conv + layer

    return layer


def batch_normalization(x, num_filters):
    epsilon = 1e-3

    shape = [num_filters]
    scale = tf.Variable(tf.ones(shape), name='scale')
    shift = tf.Variable(tf.zeros(shape), name='shift')

    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    x_normed = tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))

    return scale * x_normed + shift
