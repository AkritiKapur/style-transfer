import tensorflow as tf


def mean_squared_error(t1, t2):
    """
        Gives the mean squared error between two tensors
    :param t1: first tensor
    :param t2: second tensor
    :return: Mean squared error (Average of the squared errors)
    """
    return tf.reduce_mean(tf.square(t1 - t2))


def get_content_loss(session, model, content_image, layer_ids):
    """
    :param session: Tensorflow session
    :param model: VGG model for example,
    :param content_image: {Numpy array} Content image
    :param layer_ids: indices of the layers for which feature maps are extracted,
                      Layers selected for optimizing content loss ~ Higher level layers!
    :return:
    """
    # initialize a feed dict with content image
    feed_dict = model.create_feed_dict(image=content_image)

    # Gets references to tensors of conv filters (layers) for specified layer ids
    layers = model.get_layer_tensors(layer_ids)

    # Finds output values (feature maps) when running content image
    # through the model.
    values = session.run(layers, feed_dict=feed_dict)

    # Calculate losses for each layer in layers
    with model.graph.as_default():
        layer_losses = []

        for value, layer in zip(values, layers):
            # Get the value (feature map) when running content image is run through the model.
            value_const = tf.constant(value)

            # Supposed to calculate the loss between mixed image feature map when passed through
            # the model and the value got from above.
            # This is just a place holder for now
            loss = mean_squared_error(layer, value_const)

            layer_losses.append(loss)

        # Average loss across all layers.
        total_loss = tf.reduce_mean(layer_losses)

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


def get_style_loss(session, model, style_image, layer_ids):
    """
    :param session: Tensorflow session
    :param model: VGG model for example,
    :param style_image: {Numpy array} Style image
    :param layer_ids: indices of the layers for which feature maps are extracted,
                      Layers selected for optimizing content loss ~ Lower level layers!
    :return:
    """
    # Placeholder for feed dict
    feed_dict = model.create_feed_dict(image=style_image)

    # Gets references to tensors of conv filters (layers) for specified layer ids
    layers = model.get_layer_tensors(layer_ids)

    with model.graph.as_default():

        # Extract gram matrices for each layer
        # Gram matrix gives co-relation between different features in the feature map
        gram_layers = [get_gram_matrix(layer) for layer in layers]

        # Get the value (gram matrix) when running content image is run through the model.
        values = session.run(gram_layers, feed_dict=feed_dict)

        layer_losses = []

        for value, gram_layer in zip(values, gram_layers):

            value_const = tf.constant(value)

            # Supposed to calculate the loss between mixed image gram matrix when passed through
            # the model and the value got from above.
            # This is just a place holder for now.
            loss = mean_squared_error(gram_layer, value_const)

            layer_losses.append(loss)

        # Average loss across all layers.
        total_loss = tf.reduce_mean(layer_losses)

    return total_loss


def get_variational_loss(model):
    # Supposed to make results better.

    # Calculates sum of the pixel values between the original and the shifterd image
    # shifted image is shifted by one pixel on each axis.
    loss = tf.reduce_sum(tf.abs(model.input[:, 1:, :, :] - model.input[:, :-1, :, :])) + \
           tf.reduce_sum(tf.abs(model.input[:, :, 1:, :] - model.input[:, :, :-1, :]))

    return loss
