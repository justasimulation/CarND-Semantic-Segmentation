import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import time
import numpy as np
import project_tests as tests


LEARNING_RATE = 1e-5

# epsilon parameter of adam optimizer
ADAM_EPS = 1e-5

# this is for vgg16 dropout layer, but as it is not connected to the output this is not used actually
KEEP_PROB     = 0.5

# std for layers initialization
FC7_CONV_INIT_STD       = 1e-2
POOL4_INIT_STD          = 1e-3
POOL3_INIT_STD          = 1e-4

# regularization scale
REG_SCALE     = 5e-4

NUM_SAMPLES = 1000
NUM_EPOCHS  = 200
BATCH_SIZE  = 40
# number of labeled classes, currently: background, road
NUM_CLASSES = 2

# all images are resized to this shape
IMAGE_SHAPE = (160, 576)
DATA_DIR = './data'
TRAINING_DIR = "data_road/training"
RUNS_DIR = './runs'

VIDEO_FILE_NAME = "challenge_video.mp4"

MODEL_PATH = "model/model.ckpt"


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Loads pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, pool3_out, pool4_out, fc7_conv_out)
    where fc7_conv_out is the output of last fully connected layers of vgg16 converted into convolutional layers
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = "vgg16"
    vgg_input_tensor_name       = "image_input:0"
    vgg_keep_prob_tensor_name   = "keep_prob:0"
    vgg_pool3_out_tensor_name   = "layer3_out:0"
    vgg_pool4_out_tensor_name   = "layer4_out:0"
    vgg_fc7_conv_out            = "layer7_out:0"

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    image_input     = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob       = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    pool3_out       = graph.get_tensor_by_name(vgg_pool3_out_tensor_name)
    pool4_out       = graph.get_tensor_by_name(vgg_pool4_out_tensor_name)
    fc7_conv_out    = graph.get_tensor_by_name(vgg_fc7_conv_out)

    return image_input, keep_prob, pool3_out, pool4_out, fc7_conv_out

tests.test_load_vgg(load_vgg, tf)


def layers(pool3_out, pool4_out, fc7_conv_out, keep_prob, num_classes):
    """
    Create the layers for a fully convolutional network. Build skip-layers using the vgg layers.
    :param pool3_out: tensor output of pool3 layer
    :param pool4_out: tensor output of pool4 layer
    :param fc7_conv_out: tensor output of fc7 layer which was converted into convolutional layer
    :param keep_prob: placeholder for keep probability
    :param num_classes: Number of classes to classify
    :return: tensor for the last layer of output
    """
    # TODO: Implement function

    # apply convolution to 7th layer
    score_7 = tf.nn.dropout(tf.layers.conv2d(fc7_conv_out, filters=num_classes, kernel_size=1, padding="same",
                                             kernel_initializer=tf.random_normal_initializer(stddev=FC7_CONV_INIT_STD),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=REG_SCALE)),
                            keep_prob=keep_prob)

    # upsample convoluted 7th layer
    upsample_7x2 = tf.layers.conv2d_transpose(score_7, filters=num_classes, kernel_size=4, strides=2, padding="same",
                                              kernel_initializer=helper.create_deconv_filter(score_7, 4, num_classes),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=REG_SCALE))

    # apply convolution to 4th layer
    score_4 = tf.nn.dropout(tf.layers.conv2d(pool4_out, filters=num_classes, kernel_size=1, padding="same",
                                             kernel_initializer=tf.random_normal_initializer(stddev=POOL4_INIT_STD),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=REG_SCALE)),
                            keep_prob=keep_prob)

    # compose 4th and upsampled 7th convoluted layer
    fuse_4_7 = tf.add(score_4, upsample_7x2)

    # upsample composition result
    upsample_4_7x2 = tf.layers.conv2d_transpose(fuse_4_7, filters=num_classes, kernel_size=4, strides=2, padding="same",
                                                kernel_initializer=helper.create_deconv_filter(fuse_4_7, 4, num_classes),
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=REG_SCALE))

    # apply convolution to 3d layer
    score_3 = tf.nn.dropout(tf.layers.conv2d(pool3_out, filters=num_classes, kernel_size=1, padding="same",
                                             kernel_initializer=tf.random_normal_initializer(stddev=POOL3_INIT_STD),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=REG_SCALE)),
                            keep_prob=keep_prob)

    # compose with the previous result
    fuse_3_4_7 = tf.add(upsample_4_7x2, score_3)

    # upsample the result so it matches the original image size
    upsample_4_7_3x8 = tf.layers.conv2d_transpose(fuse_3_4_7, filters=num_classes, kernel_size=16, strides=8, padding="same",
                                                  kernel_initializer=helper.create_deconv_filter(fuse_3_4_7, 16, num_classes),
                                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=REG_SCALE))

    return upsample_4_7_3x8

#tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes, is_training):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :param is_training: Specifies whether there is a need to add regularization loss
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels_one_hot = tf.reshape(correct_label, (-1, num_classes))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_one_hot))

    if is_training:
        loss += tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=ADAM_EPS).minimize(loss)

    return logits, train_op, loss

#tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, recall, recall_op):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    :param recall: TF Tensor for recall
    :param recall_op: TF Operation for recall
    """
    # TODO: Implement function
    start_time = time.time()
    last_time = start_time

    print()
    print("Training...")
    print("Start time: {}".format(time.ctime(start_time)))

    for epoch in range(epochs):

        batch_count = 0
        sample_count = 0

        for image, label in get_batches_fn(batch_size):
            _, loss, _, r = sess.run([train_op, cross_entropy_loss, recall_op, recall],
                                      feed_dict={input_image: image,
                                                 correct_label: label,
                                                 learning_rate: LEARNING_RATE,
                                                 keep_prob: KEEP_PROB})
            cur_time = time.time()
            batch_count += 1
            sample_count += image.shape[0]

            print("Epoch: {}, Batch: {:02d}, Loss: {:.3f}, Recall: {:.3f}, Time: {:.3f}".format(epoch,
                                                                                                batch_count,
                                                                                                loss,
                                                                                                r,
                                                                                                cur_time - last_time))
            last_time = cur_time

    print("Overall time: {:.3f}".format(time.time() - start_time))

#tests.test_train_nn(train_nn)


def evaluate_nn(sess, batch_size, get_batches_fn, cross_entropy_loss, input_image,
                correct_label, logits, keep_prob, num_classes):
    """
    Evaluates the model on validation data
    :param sess: TF Session
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of validation data. Call using get_batches_fn(batch_size)
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    """

    print()
    print("Evaluating model...")

    start_time = time.time()

    # create confusion matrix
    predictions = tf.argmax(logits, 1)
    labels = tf.argmax(tf.reshape(correct_label, (-1, num_classes)), 1)

    confusion_matrix = tf.confusion_matrix(labels=labels, predictions=predictions)

    print("Start time: {}".format(time.ctime(start_time)))

    sample_count = 0

    total_background_recall = 0
    total_road_recall = 0

    min_background_recall = 1
    min_road_recall = 1

    # for all the images calculate road recall and background recall, then find average values
    for image, label in get_batches_fn(batch_size):
        loss, conf = sess.run([cross_entropy_loss, confusion_matrix],
                              feed_dict={input_image: image,
                                         correct_label: label,
                                         keep_prob: 1.})
        cur_batch_size = image.shape[0]
        sample_count += cur_batch_size

        background_recall = conf[0, 0] / np.sum(conf[0, :])
        road_recall = conf[1, 1] / np.sum(conf[1, :])

        total_background_recall += background_recall * cur_batch_size
        total_road_recall += road_recall * cur_batch_size

        min_background_recall = min(min_background_recall, background_recall)
        min_road_recall = min(min_road_recall, road_recall)

    print("Average values: background recall: {:.3f}, road recall: {:.3f}".
          format(total_background_recall/sample_count, total_road_recall/sample_count))
    print("Min values: background recall: {:.3f}, road recall: {:.3f}".
          format(min_background_recall, min_road_recall))
    print("Overall time: {:.3f}".format(time.time() - start_time))


def run():
    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(DATA_DIR)

    tests.test_for_kitti_dataset(DATA_DIR)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(DATA_DIR, 'vgg')

        # Create function to get batches

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        get_batches_fn_augmented, \
        get_batches_fn_test = helper.gen_augmented_batch_function(os.path.join(DATA_DIR, TRAINING_DIR),
                                                                  IMAGE_SHAPE, NUM_SAMPLES)

        # TODO: Build NN using load_vgg, layers, and optimize function

        labels = tf.placeholder(dtype=tf.float32, shape=[None, None, None, NUM_CLASSES])
        learning_rate = tf.placeholder(dtype=tf.float32)

        input_image, keep_prob, pool3_out, pool4_out, fc7_conv_out = load_vgg(sess, vgg_path)

        layer_output = layers(pool3_out, pool4_out, fc7_conv_out, keep_prob,NUM_CLASSES)

        # create optimization ops for training with regularization
        logits, train_op, loss = optimize(layer_output, labels, learning_rate, NUM_CLASSES, True)

        recall, recall_op = tf.metrics.recall(tf.argmax(tf.reshape(labels, (-1, NUM_CLASSES)), 1), tf.argmax(logits, 1))

        # TODO: Train NN using the train_nn function

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver()

        train_nn(sess=sess, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, get_batches_fn=get_batches_fn_augmented,
                 train_op=train_op, cross_entropy_loss=loss, input_image=input_image, correct_label=labels,
                 keep_prob=keep_prob, learning_rate=learning_rate, recall=recall, recall_op=recall_op)

        # create optimization ops for testing without regularization
        logits, train_op, loss = optimize(layer_output, labels, learning_rate, NUM_CLASSES, False)

        evaluate_nn(sess=sess, batch_size=BATCH_SIZE, get_batches_fn=get_batches_fn_test, cross_entropy_loss=loss,
                    input_image=input_image, correct_label=labels, logits=logits, keep_prob=keep_prob,
                    num_classes=NUM_CLASSES)

        # TODO: Save inference data using helper.save_inference_samples
        output_dir = helper.create_output_dir(RUNS_DIR)

        helper.save_inference_samples(output_dir, DATA_DIR, sess, IMAGE_SHAPE, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video
        helper.create_video(output_dir, DATA_DIR, VIDEO_FILE_NAME, sess, logits, keep_prob, input_image)

        # save model
        saver.save(sess, os.path.join(output_dir, MODEL_PATH))


if __name__ == '__main__':
    run()
