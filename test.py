"""
    Tests if CNN code works
    Example taken from:
    https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb
"""
import numpy as np
import tensorflow as tf
import time

from cnn_utils import get_conv_layer, get_fc_layer
from tensorflow.examples.tutorials.mnist import input_data

filter_1 = 3
num_filters_1 = 8

filter_2 = 5
num_filters_2 = 12

# Fully connected layer
fc_size = 128



def trial_conv():
    data = input_data.read_data_sets('data/MNIST/', one_hot=True)
    data.test.cls = np.argmax(data.test.labels, axis=1)

    # We know that MNIST images are 28 pixels in each dimension.
    img_size = 28

    # Images are stored in one-dimensional arrays of this length.
    img_size_flat = img_size * img_size

    # Tuple with height and width of images used to reshape arrays.
    img_shape = (img_size, img_size)

    # Number of colour channels for the images: 1 channel for gray-scale.
    num_channels = 1

    # Number of classes, one class for each of 10 digits.
    num_classes = 10

    # None as the first argument since we do not know the number of training examples while training.
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

    # Reshape x to be used in CNN
    x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

    # Create Placeholders

    # Set placeholder for true labels (one hot)
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

    # True Class labels
    y_true_class = tf.argmax(y_true, axis=1)

    # Create 1st conv layer
    conv_layer_1, filers_1 = get_conv_layer(input=x_image, n_input_channels=num_channels,
                                            filter_width=filter_1, filter_height=filter_1,
                                            n_channels=num_filters_1)

    # Create 2nd conv layer
    conv_layer_2, filers_2 = get_conv_layer(input=conv_layer_1, n_input_channels=num_filters_1,
                                            filter_width=filter_2, filter_height=filter_2,
                                            n_channels=num_filters_2)


    num_features = conv_layer_2.get_shape()[1:4].num_elements()
    layer_flat = tf.reshape(conv_layer_2, [-1, num_features])

    fc_1 = get_fc_layer(input=layer_flat, n_input=num_features, n_output=fc_size)
    fc_2 = get_fc_layer(input=fc_1, n_input=fc_size, n_output=num_classes)

    y_pred = tf.nn.softmax(fc_2)

    y_pred_label = tf.argmax(y_pred, axis=1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc_2, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    correct_prediction = tf.equal(y_pred_label, y_true_class)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create tensorflow session
    session = tf.Session()

    session.run(tf.initialize_all_variables())

    optimize(num_iterations=61, data=data, session=session, optimizer=optimizer, x=x, y_true=y_true, accuracy=accuracy)


def optimize(num_iterations, data, session, optimizer, x, y_true, accuracy):
    total_iterations = 0
    train_batch_size = 96

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 10 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations


if __name__ == '__main__':
    trial_conv()
