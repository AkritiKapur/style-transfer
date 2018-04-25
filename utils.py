import numpy as np
import PIL.Image

# Define Image Utils
def get_numpy_image():
    pass


def view_image():
    pass

import tensorflow as tf


# Define Deep Learning Utils

# Loss Utils
def mean_squared_error(t1, t2):
    """
        Gives the mean squared error between two tensors
    :param t1: first tensor
    :param t2: second tensor
    :return: Mean squared error (Average of the squared errors)
    """
    return tf.reduce_mean(tf.square(t1 - t2))


def get_content_loss():
    pass


def get_style_loss():
    pass
