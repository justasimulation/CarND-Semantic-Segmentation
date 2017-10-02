import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm

from moviepy.editor import VideoFileClip


# defines boundary for image crop for augmentation,
# e.g. from 0 to 0.25 of side length
CROP_COEFF = 0.25

# pixel intencities will be multiplied by this number to emulate shadows
HALF_DARKEN_COEFF = 0.3

# augmentation transformation probabilities
HALF_DARKEN_PROB = 0.7
CROP_PROB = 0.7
FLIP_PROB = 0.5


# background is marked with this color
BACKGROUND_COLOR = [255, 0, 0]
# roads are marked with this color
ROAD_COLOR       = [255, 0, 255]

TEST_SET_SHARE = 0.10


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def preprocess_image(image):
    """
    Performs image preprocessing.
    :param image:
    :return:
    """
    return image.astype(np.float32) - [123.68, 103.939, 116.779]


def random_change_image(image, label):
    """
    Performs random image change needed for augmentation. In case of resizing, label is resized as well.
    :param image:
    :param label:
    :return: image, label
    """
    if random.random() < HALF_DARKEN_PROB:
        left, right = (0, image.shape[1] // 2) if random.random() > 0.5 else (image.shape[1] // 2, image.shape[1])
        image[:, left:right] = (image[:, left:right] * HALF_DARKEN_COEFF)

    if random.random() < CROP_PROB:
        left    = int(random.random() * CROP_COEFF * image.shape[1])
        right   = int((1 - random.random() * CROP_COEFF) * image.shape[1])
        top     = int(random.random() * CROP_COEFF * image.shape[0])
        bottom  = int((1 - random.random() * CROP_COEFF) * image.shape[0])
        image = image[top:bottom, left:right]
        label = label[top:bottom, left:right]

    if random.random() < FLIP_PROB:
        image = np.fliplr(image)
        label = np.fliplr(label)

    return image, label


def get_images(image_paths, label_paths, batch_size, image_shape, num_samples, augment_fn=None):
    """
    Loads image and label. Transforms label into expected format. Transforms image using augment_fn if needed.
    Resizes label and image into needed shape.
    :param image_paths: list of paths to images
    :param label_paths: dict [image_file_name: label file path]
    :param batch_size: batch size
    :param image_shape: all images will be resized to this shape
    :param num_samples: number of samples in one epoch, this function will generate this number of samples overall
    :return: generator that returns (image, label)
    """
    for batch_i in range(0, num_samples, batch_size):
        images = []
        label_images = []
        for i in range(batch_i, min(batch_i + batch_size, num_samples)):
            # read images
            idx = i % len(image_paths)

            image = scipy.misc.imread(image_paths[idx])
            label_image = scipy.misc.imread(label_paths[os.path.basename(image_paths[idx])])

            # in case agumentaion is needed
            if augment_fn is not None:
                image, label_image = augment_fn(image, label_image)

            # resize
            image = scipy.misc.imresize(image, image_shape)
            label_image = scipy.misc.imresize(label_image, image_shape, interp="nearest")

            # get road pixels
            gt_road = np.all(label_image == ROAD_COLOR, axis=2)
            gt_road = gt_road.reshape(*gt_road.shape, 1)

            # background is everything else except the road
            gt_bg = np.invert(gt_road)

            # combine all 2d planes together so they contain one hot vectors along depth axis
            gt_image = np.concatenate((gt_bg, gt_road), axis=2)

            images.append(preprocess_image(image))
            label_images.append(gt_image)

        yield np.array(images), np.array(label_images)


def gen_augmented_batch_function(data_folder, image_shape, requested_images_num):
    """
    Generate functions to create batches of augmented training data and not agumented validation data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :param requested_images_num: Need to generate new images so the overall images number equals to this value.
    :return: training generator function, validation generator function
    """

    # enumerate images and labels
    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
    label_paths = {
        re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
        for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}

    random.shuffle(image_paths)

    # split into test and valid sets
    valid_len = int(len(image_paths) * TEST_SET_SHARE)
    valid_paths = image_paths[:valid_len]
    train_paths = image_paths[valid_len:]

    overall_images_num = max(requested_images_num, len(train_paths))

    print()
    print("Number of training images: {}\nNumber of validation images: {}".format(len(train_paths), len(valid_paths)))
    print("Number of images per epoch including augmented ones: {}".format(overall_images_num))

    def get_batches_fn_valid(batch_size):

        return get_images(valid_paths, label_paths, batch_size, image_shape, len(valid_paths))

    def get_batches_fn(batch_size):

        random.shuffle(train_paths)

        return get_images(train_paths, label_paths, batch_size, image_shape, overall_images_num, random_change_image)

    return get_batches_fn, get_batches_fn_valid


def gen_marked_image(sess, logits, keep_prob, image_pl, image, image_shape):
    """
    Draws main road with green and side road with blue on the given image
    :param sess: tf session
    :param logits: logits variable to calculate predictions
    :param keep_prob: keep probability placeholder
    :param image_pl: image placeholder
    :param image: image
    :param image_shape: image shape
    :return: np.array of the same shape as image with drawn main and side roads
    """
    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, image_pl: [preprocess_image(image)]})

    im_softmax = np.argmax(im_softmax[0], axis=1).reshape(image_shape[0], image_shape[1])
    street_im = scipy.misc.toimage(image)

    segmentation = (im_softmax == 1).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im.paste(mask, box=None, mask=mask)

    return np.asarray(street_im.getdata()).reshape(image_shape[0], image_shape[1], len(street_im.getbands()))


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        street_im = gen_marked_image(sess, logits, keep_prob, image_pl, image, image_shape)

        yield os.path.basename(image_file), np.array(street_im)


def create_output_dir(runs_dir):
    """
    Creates a directory for a particular run in common runs dir
    :param runs_dir: common runs dir
    :return: particular run directory
    """
    return os.path.join(runs_dir, str(time.time()))


def save_inference_samples(output_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    """
    Marks road pixels on all images in the given directory.
    :param output_dir: where to save processed images
    :param data_dir: where to load images for processing
    :param sess: tf session
    :param image_shape: all the images will be resized to this shape
    :param logits: logits tf variable
    :param keep_prob: placeholder
    :param input_image: placeholder
    :return:
    """
    # Make folder for current run
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)


def get_32_image_shape(image):
    """
    Returns image shape so to is can be divided by 32 without remainder. It is needed so the result of the network
    would of the same shape as its input.
    :param image: nd.array of shape (height, width, depth)
    :return: (height, width)
    """
    return int(image.shape[0] / 32.) * 32, int(image.shape[1] / 32.) * 32


def process_video(source_file_name, dest_file_name, sess, logits, keep_prob, image_pl):
    """
    Processes each frame of given video file and marks road pixels.
    :param source_file_name:
    :param dest_file_name:
    :param sess: tf session
    :param logits: logits tf variable
    :param keep_prob: placeholder
    :param image_pl: image placeholder
    """
    clip = VideoFileClip(filename=source_file_name)
    processed_clip = clip.fl_image(lambda image: gen_marked_image(sess, logits, keep_prob, image_pl,
                                                                  scipy.misc.imresize(image, get_32_image_shape(image)),
                                                                  get_32_image_shape(image)))
    processed_clip.write_videofile(dest_file_name, audio=False)


def create_video(output_dir, data_dir, source_file_name, sess, logits, keep_prob, input_image):
    """
    Creates video file with marked road pixels
    :param output_dir: output dir
    :param data_dir:  common data dir
    :param source_file_name: video file name
    :param sess: tf session
    :param logits: logits tf variable
    :param keep_prob: placeholder
    :param input_image: placeholder
    """
    dest_dir = os.path.join(output_dir, "video")
    os.makedirs(dest_dir)
    dest_file_name = "processed_{}".format(source_file_name)

    source_dir = os.path.join(data_dir, "video")

    process_video(os.path.join(source_dir, source_file_name), os.path.join(dest_dir, dest_file_name),
                  sess, logits, keep_prob, input_image)


def create_deconv_filter(input, ksize, num_filters):
    """
    Creates weights for bilinear upsampling using transposed convolution.
    Copyright https://github.com/MarvinTeichmann
    :param input: input tensor
    :param ksize: kernel size
    :param num_filters: number of output filters
    :return:
    """
    f_shape = [ksize, ksize, num_filters, input.get_shape()[3].value]
    width = f_shape[0]
    heigh = f_shape[0]
    f = np.ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    return tf.constant_initializer(value=weights, dtype=tf.float32)

