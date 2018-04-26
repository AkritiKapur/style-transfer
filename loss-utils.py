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


def get_style_loss():
    pass