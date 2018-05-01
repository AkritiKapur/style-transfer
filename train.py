import numpy as np
import tensorflow as tf

from loss_utils import get_content_loss, get_style_loss, get_variational_loss, get_gram_matrix
from image_utils import plot_images, view_image, get_numpy_image, get_image, get_images, list_images
from johnson_img_transform import image_transformation_network
from vgg19 import VGG, preprocess


VGG_PATH  = './imagenet-vgg-19-weights.npz'
SAVE_PATH = 'checkpoints/model.ckpt'


def get_np_gram(t):
    shape = t.shape

    num_channels = int(shape[3])

    # flatten the image array for all channels
    # Necessarily normalizes the matrix
    matrix = np.reshape(t, (-1, num_channels))

    # Multiply the transpose of the matrix with itself to get gram matrix
    gram_matrix = np.matmul(matrix.T, matrix)

    return gram_matrix


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
        style = tf.placeholder(tf.float32, shape=style_shape, name='style_image')

        # pass style image through 'pretrained VGG-19 network'
        style_img_preprocess = preprocess(style)
        style_net = model.forward(style_img_preprocess)

        # Extract gram matrices for style image, for each layer
        gram_style = {layer: get_np_gram(style_net[layer].eval(feed_dict={style: style_image}))
                      for layer in style_layer_ids}

    # compute perceptual losses
    with tf.Graph().as_default(), tf.Session() as sess:
        # Get content image by passing it through the transformation net
        content = tf.placeholder(tf.float32, shape=content_image.shape, name='content_images')
        transformed_content = image_transformation_network(content_image=content)

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
        loss_denoise = get_variational_loss(transformed_content)

        loss_combined = weight_content * loss_content + \
                        weight_style * loss_style + \
                        weight_denoise * loss_denoise

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        grads = optimizer.compute_gradients(loss_combined)
        out = optimizer.apply_gradients(grads)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

        for i in range(num_iterations):

            sess.run(out, feed_dict={content: content_image})

            # Display status once every 10 iterations, and the last.
            if (i % 10 == 0) or (i == num_iterations - 1):
                saver.save(sess, SAVE_PATH, global_step=i)
                print()
                print("Iteration:", i)
                to_get = [loss_style, loss_content, loss_combined]

                tup = sess.run(to_get, feed_dict={content: content_image})
                _loss_style, _loss_content, _loss_combined = tup
                losses = (_loss_style, _loss_content, _loss_combined)
                print(losses)

    return True

if __name__ == '__main__':
    CONTENT_LAYER = ['relu4_2']
    STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

    content_filename = 'data/test_content'
    content_images = list_images(content_filename)
    content_image = get_images(content_images, 256, 256)

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
                         num_iterations=60,
                         step_size=10.0)
