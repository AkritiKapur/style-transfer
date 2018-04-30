import numpy as np
import tensorflow as tf

from loss_utils import get_content_loss, get_style_loss, get_variational_loss
from image_utils import plot_images, view_image, get_numpy_image, get_image
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

    # compute perceptual losses
    with tf.Graph().as_default(), tf.Session() as sess:
        # Get content image by passing it through the transformation net
        transformed_content = image_transformation_network(content_image=content_image)

        # pass tranformed content through the model
        mixed_preprocess = preprocess(transformed_content)
        mixed_net = model.forward(mixed_preprocess)

        # Create the loss-function for
        loss_content = get_content_loss(model=model,
                                        content_image=content_image,
                                        layer_ids=content_layer_ids,
                                        mixed_net=mixed_net)

        # Create the loss-function for the style-layers and -image.
        loss_style = get_style_loss(model=model,
                                    style_image=style_image,
                                    layer_ids=style_layer_ids,
                                    mixed_net=mixed_net)

        # Create the loss-function for the denoising of the mixed-image.
        loss_denoise = get_variational_loss(model)

    # adjust levels of loss functions, normalize them
    # multiply them with a variable
    # taking reciprocal values of loss values of content, style, denoising
    # small constant to avoid divide by 0
    # adjustment value normalizes loss so approximately 1
    # weights should be set relative to each other dont depend on layers
    # we are using

    # Create TensorFlow variables for adjusting the values of
    # the loss-functions. This is explained below.
    adj_content = tf.Variable(1e-10, name='adj_content')
    adj_style = tf.Variable(1e-10, name='adj_style')
    adj_denoise = tf.Variable(1e-10, name='adj_denoise')

    # Initialize the adjustment values for the loss-functions.
    session.run([adj_content.initializer,
                 adj_style.initializer,
                 adj_denoise.initializer])

    # Create TensorFlow operations for updating the adjustment values.
    # These are basically just the reciprocal values of the
    # loss-functions, with a small value 1e-10 added to avoid the
    # possibility of division by zero.
    update_adj_content = adj_content.assign(1.0 / (loss_content + 1e-10))
    update_adj_style = adj_style.assign(1.0 / (loss_style + 1e-10))
    update_adj_denoise = adj_denoise.assign(1.0 / (loss_denoise + 1e-10))

    # This is the weighted loss-function that we will minimize
    # below in order to generate the mixed-image.
    # Because we multiply the loss-values with their reciprocal
    # adjustment values, we can use relative weights for the
    # loss-functions that are easier to select, as they are
    # independent of the exact choice of style- and content-layers.
    loss_combined = weight_content * adj_content * loss_content + \
                    weight_style * adj_style * loss_style + \
                    weight_denoise * adj_denoise * loss_denoise

    # Use TensorFlow to get the mathematical function for the
    # gradient of the combined loss-function with regard to
    # the input image. (mixed)
    gradient = tf.gradients(loss_combined, model.input)
    feed_dict = model.create_feed_dict(image=mixed_image)

    # # List of tensors that we will run in each optimization iteration.
    # run_list = [gradient, update_adj_content, update_adj_style,
    #             update_adj_denoise]

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss_combined)

    for i in range(num_iterations):
        # Create a feed-dict with the mixed-image.
        feed_dict = model.create_feed_dict(image=content_image)

        # Use TensorFlow to calculate the value of the
        # gradient, as well as updating the adjustment values.
        # grad, adj_content_val, adj_style_val, adj_denoise_val \
        #     = session.run(run_list, feed_dict=feed_dict)

        # Reduce the dimensionality of the gradient.
        # Remove single-dimensional entries from the shape of an array.
        # grad = np.squeeze(grad)

        # Scale the step-size according to the gradient-values.
        # Ratio of weights:updates
        # akin to learning rate
        # step_size_scaled = step_size / (np.std(grad) + 1e-8)

        # Update the image by following the gradient.
        # gradient descent
        # mixed_image -= grad * step_size_scaled

        optimizer.run(feed_dict=feed_dict)
        # Ensure the image has valid pixel-values between 0 and 255.
        # Given an interval, values outside the interval are clipped
        # to the interval edges.
        # mixed_image = np.clip(mixed_image, 0.0, 255.0)

        # Print a little progress-indicator.
        print(". ", end="")

        # Display status once every 10 iterations, and the last.
        if (i % 10 == 0) or (i == num_iterations - 1):
            print()
            print("Iteration:", i)
            to_get = [loss_style, loss_content, loss_denoise, loss_combined]
            test_feed_dict = model.create_feed_dict(image=content_image)

            tup = session.run(to_get, feed_dict=test_feed_dict)
            _loss_style, _loss_content, _loss_denoise, _loss_combined = tup
            losses = (_loss_style, _loss_content, _loss_denoise, _loss_combined)
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

    print()
    print("Final image:")
    view_image(mixed_image)

    # Close the TensorFlow session to release its resources.
    session.close()

    # Return the mixed-image.
    return mixed_image

if __name__ == '__main__':
    CONTENT_LAYER = ['relu4_2']
    STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

    content_filename = 'images/willy_wonka_old.jpg'
    content_image = get_image(content_filename, max_size=256)

    style_filename = 'images/style5.jpg'
    style_image = get_image(style_filename, max_size=256)
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
