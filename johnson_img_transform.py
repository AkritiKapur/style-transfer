import numpy as np
import tensorflow as tf

from cnn_utils import get_conv_layer, get_res_layer, get_deconv_layer


def image_transformation_network(content_image):
    n_content = int(content_image.shape[0])
    content_image = content_image / 127.5 - 1
    conv1, _ = get_conv_layer(content_image, filter_height=9, filter_width=9, n_channels=32, n_input_channels=3)
    conv2, _ = get_conv_layer(conv1, filter_height=3, filter_width=3, n_channels=64,
                              n_input_channels=32, stride=[1, 2, 2, 1])
    conv3, _ = get_conv_layer(conv2, filter_height=3, filter_width=3, n_channels=128,
                              n_input_channels=64, stride=[1, 2, 2, 1])
    res1 = get_res_layer(conv3, filter_height=3, filter_width=3, n_filters=128, n_input_channels=128)
    res2 = get_res_layer(res1, filter_height=3, filter_width=3, n_filters=128, n_input_channels=128)
    res3 = get_res_layer(res2, filter_height=3, filter_width=3, n_filters=128, n_input_channels=128)
    res4 = get_res_layer(res3, filter_height=3, filter_width=3, n_filters=128, n_input_channels=128)
    res5 = get_res_layer(res4, filter_height=3, filter_width=3, n_filters=128, n_input_channels=128)
    conv4, _ = get_deconv_layer(res5, filter_height=3, filter_width=3, n_input_channels=128,
                                n_output_channels=(n_content, 128, 128, 64), stride=[1, 2, 2, 1])
    conv5, _ = get_deconv_layer(conv4, filter_height=3, filter_width=3, n_output_channels=(n_content, 256, 256, 32),
                                n_input_channels=64, stride=[1, 2, 2, 1])
    conv6, _ = get_conv_layer(conv5, filter_height=9, filter_width=9, n_channels=3, n_input_channels=32, relu=False)
    conv6 = tf.tanh(conv6)

    tf.verify_tensor_all_finite(conv6, msg="conv 6")
    output = (conv6 + 1) * 127.5

    return output
