import tensorflow as tf
from vgg19 import preprocess

input_shape = (256, 256, 3)


def mean_squared_error(t1, t2):
    """
        Gives the mean squared error between two tensors
    :param t1: first tensor
    :param t2: second tensor
    :return: Mean squared error (Average of the squared errors)
    """
    return tf.reduce_mean(tf.square(t1 - t2)) + 0.00001


def get_content_loss(model, content_image, layer_ids, mixed_net):
    """
    :param model: VGG model for example,
    :param content_image: {Numpy array} Content image
    :param layer_ids: indices of the layers for which feature maps are extracted,
                      Layers selected for optimizing content loss ~ Higher level layers!
    :param mixed_image: Stylized image, image the loss is computed against.
    :return:
    """
    # Pass content_images through 'pretrained VGG-19 network'
    content_imgs_preprocess = preprocess(content_image)
    content_net = model.forward(content_imgs_preprocess)

    layer_losses = []
    for layer in layer_ids:
        # Calculate loss between the mixed image and the content image
        loss = 2 * tf.nn.l2_loss(content_net[layer] - mixed_net[layer]) / tf.to_float(tf.size(mixed_net[layer]))
        # loss = mean_squared_error(content_net[layer], mixed_net[layer])

        layer_losses.append(loss)

    # Average loss across all layers.
    total_loss = tf.reduce_mean(layer_losses)

    # print(total_loss.eval())
    return total_loss


def get_gram_matrix(t):
    """
    :param t: Input tensor example a conv layer tensor
    :return: Gram matrix of the tensor (Basically a product of matrix and its transpose)
    """

    # Shape 4D because image pixels height * width * RGB * number-filters-in-layer
    shape = t.get_shape()

    num_channels = int(shape[3])

    # flatten the image array for all channels
    # Necessarily normalizes the matrix
    matrix = tf.reshape(t, shape=[-1, num_channels])

    # Multiply the transpose of the matrix with itself to get gram matrix
    gram_matrix = tf.matmul(tf.transpose(matrix), matrix)

    return gram_matrix


def get_style_loss(model, gram_layers_style, layer_ids, mixed_net):
    """
    :param session: Tensorflow session
    :param model: VGG model for example,
    :param style_image: {Numpy array} Style image
    :param layer_ids: indices of the layers for which feature maps are extracted,
                      Layers selected for optimizing content loss ~ Lower level layers!
    :return:
    """

    # Extract gram matrices for mixed image, for each layer
    gram_layers_mixed = {layer: get_gram_matrix(mixed_net[layer]) for layer in layer_ids}

    layer_losses = []

    # Calculate loss between the mixed image and the style image gram matrices
    for layer in layer_ids:
        style = gram_layers_style[layer]
        mixed = gram_layers_mixed[layer]

        loss = 2 * tf.nn.l2_loss(mixed - style) / tf.to_float(tf.size(style))
        # loss = mean_squared_error(style, mixed)

        layer_losses.append(loss)

    # Average loss across all layers.
    total_loss = tf.reduce_mean(layer_losses)

    return total_loss


def get_variational_loss(mixed_image):
    # Supposed to make results better.

    # Calculates sum of the pixel values between the original and the shifted image
    # shifted image is shifted by one pixel on each axis.
    loss = tf.reduce_sum(tf.abs(mixed_image[:, 1:, :, :] - mixed_image[:, :-1, :, :])) + \
           tf.reduce_sum(tf.abs(mixed_image[:, :, 1:, :] - mixed_image[:, :, :-1, :]))

    return loss
