import numpy as np
import tensorflow as tf

from loss_utils import get_content_loss, get_style_loss, get_variational_loss, get_gram_matrix
from image_utils import plot_images, view_image, get_numpy_image, get_image, get_images
from johnson_img_transform import image_transformation_network
from vgg19 import VGG, preprocess

VGG_PATH  = './imagenet-vgg-19-weights.npz'


def style_transfer(content_image, style_image,
                   content_layer_ids, style_layer_ids,
                   vgg_path=VGG_PATH,
                   weight_content=1.5, weight_style=10.0,
                   weight_denoise=0.3,
                   num_iterations=120, step_size=10.0):
    """
    Performs SGD to minimize content, style and variational noise to
    get a stylized image that keeps the styles and textures from the style image

    :param content_image: 3D numpy array containing pixel values of content image
    :param style_image: 3D numpy array containing pixel values of style image
    :param content_layer_ids: Layer ids of content image we choose to optimize over
    :param style_layer_ids: Layer ids of style image we choose to optimize over
    :param weight_content: Weight of content loss
    :param weight_style: Weight of style loss
    :param weight_denoise: Weight of denoised noise
    :param num_iterations: Number of iterations
    :param step_size: Step size for gradient
    """

    # Assign a pretrained VGG network as the model
    model = VGG(vgg_path)

    gram_style = {}
    with tf.Graph().as_default(), tf.Session() as sess:
        style_shape = style_image.shape
        # Create a placeholder for style image
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')

        # pass style image through 'pretrained VGG-19 network'
        style_img_preprocess = preprocess(style_image)
        style_net = model.forward(style_img_preprocess)

        # Extract gram matrices for style image, for each layer
        gram_style = {layer: get_gram_matrix(style_net[layer]) for layer in style_layer_ids}

    # compute perceptual losses
    with tf.Graph().as_default(), tf.Session() as sess:
        # Get content image by passing it through the transformation net
        transformed_content = image_transformation_network(content_image=content_image)
        content = tf.placeholder(tf.float32, shape=content_image.shape, name='content_images')

        # pass tranformed content through the model
        mixed_preprocess = preprocess(transformed_content)
        mixed_net = model.forward(mixed_preprocess)

        # Create the loss-function for
        loss_content = get_content_loss(model=model,
                                        content_image=content,
                                        layer_ids=content_layer_ids,
                                        mixed_net=mixed_net)

        # Create the loss-function for the style-layers and -image.
        loss_style = get_style_loss(model=model,
                                    gram_layers_style=gram_style,
                                    layer_ids=style_layer_ids,
                                    mixed_net=mixed_net)

        # Create the loss-function for the denoising of the mixed-image.
        # loss_denoise = get_variational_loss(model)

        loss_combined = weight_content * loss_content + \
                        weight_style * loss_style
                        # weight_denoise * loss_denoise

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss_combined)
        sess.run(tf.global_variables_initializer())

        for i in range(num_iterations):

            sess.run(optimizer, feed_dict={content: content_image})

            # Display status once every 10 iterations, and the last.
            if (i % 10 == 0) or (i == num_iterations - 1):
                print()
                print("Iteration:", i)
                to_get = [loss_style, loss_content, loss_combined]

                tup = sess.run(to_get, feed_dict={content: content_image})
                _loss_style, _loss_content, _loss_combined = tup
                losses = (_loss_style, _loss_content, _loss_combined)
                print(losses)

                # Print adjustment weights for loss-functions.
                # msg = "Weight Adj. for Content: {0:.2e}, Style: {1:.2e}, Denoise: {2:.2e}"
                # print(msg.format(adj_content_val, adj_style_val, adj_denoise_val))

                # in larger resolution
                # Plot the content-, style- and mixed-images.
                # plot_images(content_image=content_image,
                #             style_image=style_image,
                #             mixed_image=mixed_image)
                # view_image(mixed_image)

    # print()
    # print("Final image:")
    # view_image(mixed_image)
    #
    # # Close the TensorFlow session to release its resources.
    # session.close()
    #
    # # Return the mixed-image.
    # return mixed_image
    return True

if __name__ == '__main__':
    CONTENT_LAYER = ['relu4_2']
    STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

    content_filename = 'images/willy_wonka_old.jpg'
    content_image = get_image(content_filename, max_size=256)

    style_filename = 'images/styles/rain_princess.jpg'
    style_image = get_images(style_filename, 256, 256)
    content_layer_ids = [4]
    style_layer_ids = list(range(13))
    img = style_transfer(content_image=content_image,
                         style_image=style_image,
                         content_layer_ids=CONTENT_LAYER,
                         style_layer_ids=STYLE_LAYERS,
                         weight_content=1.5,
                         weight_style=10.0,
                         weight_denoise=0.3,
                         num_iterations=30,
                         step_size=10.0)
