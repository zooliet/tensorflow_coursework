#!/usr/bin/env python

import sys
sys.path.append('./')
sys.path.append('../../')

from lab_utils import (
    tf, os, np, plt, logger, ap, BooleanAction,
    debug, toc, auto_increment
)

ap.add_argument('--epochs', type=int, default=10, help='number of epochs: 10*')
ap.add_argument('--batch', type=int, default=64, help='batch size: 64*')
args, extra_args = ap.parse_known_args()
logger.info(args)
# logger.info(extra_args)

if args.all:
    args.step = 0 # forced to 0

if args.debug:
    import pdb
    import rlcompleter
    pdb.Pdb.complete=rlcompleter.Completer(locals()).complete
    # import code
    # code.interact(local=locals())
    debug = breakpoint

import time
import pandas as pd

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Concatenate


### TOC
if args.step == 0:
    toc(__file__)


if args.step or args.all:
    if not os.path.exists('tmp/tf2_t0305/'):
        os.mkdir('tmp/tf2_t0305/') 


args.step = auto_increment(args.step, args.all)
### Step #1 - tf.train.Example: Data types for tf.train.Example
if args.step >= 1: 
    print("\n### Step #1 - tf.train.Example: Data types for tf.train.Example")

    # The following functions can be used to convert a value to a type compatible
    # with tf.train.Example.

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    if args.step == 1:
        logger.info("b'test_string to tf.train.Feature():")
        print(_bytes_feature(b'test_string'))

        logger.info("u'test_bytes'.enocde('utf-8') to tf.train.Feature():")
        print(_bytes_feature(u'test_bytes'.encode('utf-8')))

        logger.info("np.exp(1) to tf.train.Feature():")
        print(_float_feature(np.exp(1)))

        logger.info("True to tf.train.Feature():")
        print(_int64_feature(True))

        logger.info("1 to tf.train.Feature():")
        print(_int64_feature(1))
        
        # All proto messages can be serialized to a binary-string
        logger.info("proto messages to a binary-string")
        feature = _float_feature(np.exp(1))
        print(feature.SerializeToString())


args.step = auto_increment(args.step, args.all)
### Step #2 - tf.train.Example: Creating a tf.train.Example message
if args.step >= 2 and args.step < 8:
    print("\n### Step #2 - tf.train.Example: Creating a tf.train.Example message")

    def serialize_example(feature0, feature1, feature2, feature3):
        """
        Creates a tf.train.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.train.Example-compatible
        # data type.
        feature = {
            'feature0': _int64_feature(feature0),
            'feature1': _int64_feature(feature1),
            'feature2': _bytes_feature(feature2),
            'feature3': _float_feature(feature3),
        }

        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    # This is an example observation from the dataset.
    if args.step == 2:
        logger.info('(data => tf.Train.Feature)+ => tf.train.Example => Serialized:')
        serialized_example = serialize_example(False, 4, b'goat', 0.9876)
        print(serialized_example)

        logger.info('Serialized string => tf.train.Example:')
        example_proto = tf.train.Example.FromString(serialized_example)
        print(example_proto)


args.step = auto_increment(args.step, args.all)
### Step #3 - TFRecords format details
if args.step == 3: 
    print("\n### Step #3 - TFRecords format details")

    format_str = """ 
    uint64 length
    uint32 masked_crc32_of_length
    byte   data[length]
    uint32 masked_crc32_of_data """
    print(format_str)


args.step = auto_increment(args.step, args.all)
### Step #4 - TFRecord files using tf.data: Writing a TFRecord file
if args.step in [4, 5, 6, 7]: 
    print("\n### Step #4 - TFRecord files using tf.data: Writing a TFRecord file")

    # The number of observations in the dataset.
    n_observations = int(1e4)

    # Boolean feature, encoded as False or True.
    feature0 = np.random.choice([False, True], n_observations)

    # Integer feature, random from 0 to 4.
    feature1 = np.random.randint(0, 5, n_observations)

    # String feature
    strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
    feature2 = strings[feature1]

    # Float feature, from a standard normal distribution
    feature3 = np.random.randn(n_observations)

    features_dataset = tf.data.Dataset.from_tensor_slices(
        (feature0, feature1, feature2, feature3)
    )
    if args.step == 4:
        logger.info(f'features_dataset:\n{features_dataset}\n')

        logger.info('features_dataset.take(1):')
        for f0,f1,f2,f3 in features_dataset.take(1):
            print(f0)
            print(f1)
            print(f2)
            print(f3)
            print()

    def tf_serialize_example(f0,f1,f2,f3):
        tf_string = tf.py_function(
            serialize_example,
            (f0,f1,f2,f3),  # pass these args to the above function.
            tf.string # the return type is `tf.string`.
        )     
        return tf.reshape(tf_string, ()) # The result is a scalar

    if args.step == 4:
        example_binary_string = tf_serialize_example(f0,f1,f2,f3)
        logger.info('(f0,f1,f2,f3) => tf.train.Example => Serialized:') 
        print(example_binary_string)
        print()

    serialized_features_dataset = features_dataset.map(tf_serialize_example)
    if args.step == 4:
        logger.info('features_dataset => tf.train.Example => Serialized:') 
        for serialized_features_data in serialized_features_dataset.take(2):
            print(serialized_features_data)
            print()

    def generator():
        for features in features_dataset:
            yield serialize_example(*features)

    serialized_features_dataset = tf.data.Dataset.from_generator(
        generator, output_types=tf.string, output_shapes=()
    )
    if args.step == 4:
        logger.info('features_dataset => generator(tf.train.Example => Serialized):') 
        for serialized_features_data in serialized_features_dataset.take(2):
            print(serialized_features_data)

    # write them to a TFRecord file
    filename = 'tmp/tf2_t0305/test.tfrecord'
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(serialized_features_dataset)


args.step = auto_increment(args.step, args.all)
### Step #5 - TFRecord files using tf.data: Reading a TFRecord file
if args.step in [5, 6, 7]: 
    print("\n### Step #5 - TFRecord files using tf.data: Reading a TFRecord file")

    filename = 'tmp/tf2_t0305/test.tfrecord'
    filenames = [filename]
    raw_dataset = tf.data.TFRecordDataset(filenames)

    if args.step == 5:
        logger.info('tf.data.TFRecordDataset(filenames):')
        for raw_record in raw_dataset.take(2):
            print(repr(raw_record))
            print()

    # Create a description of the features.
    feature_description = {
        'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.train.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    parsed_dataset = raw_dataset.map(_parse_function)
    if args.step == 5:
        logger.info('tf.data.TFRecordDataset() => tf.io.parse_sigle_example() => features')
        for parsed_record in parsed_dataset.take(2):
            print(repr(parsed_record))
            print()


args.step = auto_increment(args.step, args.all)
### Step #6 - TFRecord files in Python: Writing a TFRecord file
if args.step == 6: 
    print("\n### Step #6 - TFRecord files in Python: Writing a TFRecord file")

    filename = 'tmp/tf2_t0305/test6.tfrecord'
    # Write the `tf.train.Example` observations to the file.
    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(n_observations):
            example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
            writer.write(example)

    print()
    logger.info(f"du -sh {filename}")
    os.system(f"du -sh {filename}")


args.step = auto_increment(args.step, args.all)
### Step #7 - TFRecord files in Python: Reading a TFRecord file
if args.step == 7: 
    print("\n### Step #7 - TFRecord files in Python: Reading a TFRecord file")

    filename = 'tmp/tf2_t0305/test6.tfrecord'
    filenames = [filename]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    logger.info(raw_dataset)

    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)

    
args.step = auto_increment(args.step, args.all)
### Step #8 - Walkthrough: Reading and writing image data: Fetch the images
if args.step >= 8: 
    print("\n### Step #8 - Walkthrough: Reading and writing image data: Fetch the images")

    cat_in_snow  = tf.keras.utils.get_file(
        '320px-Felis_catus-cat_on_snow.jpg', 
        'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg'
    )
    williamsburg_bridge = tf.keras.utils.get_file(
        '194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg',
        'https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg'
    )

    if args.step == 8 and args.plot:
        plt.figure()
        ax = plt.subplot(1, 2, 1)
        cat = tf.io.read_file(cat_in_snow)
        cat = tf.image.decode_jpeg(cat)
        plt.imshow(cat)
        ax = plt.subplot(1, 2, 2)
        bridge = tf.io.read_file(williamsburg_bridge)
        bridge = tf.image.decode_jpeg(bridge)
        plt.imshow(bridge)
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #9 - Walkthrough: Reading and writing image data: Write the TFRecord data
if args.step == 9: 
    print("\n### Step #9 - Walkthrough: Reading and writing image data: Write the TFRecord data")

    image_labels = {
        cat_in_snow : 0,
        williamsburg_bridge : 1,
    }

    # Create a dictionary with features that may be relevant.
    def image_example(image_string, label):
        image_shape = tf.image.decode_jpeg(image_string).shape
        feature = {
            'height': _int64_feature(image_shape[0]),
            'width': _int64_feature(image_shape[1]),
            'depth': _int64_feature(image_shape[2]),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_string),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))

    # This is an example, just using the cat image.
    image_string = open(cat_in_snow, 'rb').read()
    label = image_labels[cat_in_snow]

    if args.step == 9:
        for line in str(image_example(image_string, label)).split('\n')[:15]:
            print(line)
        print('...')

    # Write the raw image files to `images.tfrecords`.
    # First, process the two images into `tf.train.Example` messages.
    # Then, write to a `.tfrecords` file.
    record_file = 'tmp/tf2_t0305/images.tfrecords'
    with tf.io.TFRecordWriter(record_file) as writer:
        for filename, label in image_labels.items():
            image_string = open(filename, 'rb').read()
            tf_example = image_example(image_string, label)
            writer.write(tf_example.SerializeToString())

    if args.step == 9:
        logger.info(f'du -sh {record_file}')
        os.system(f"du -sh {record_file}")


args.step = auto_increment(args.step, args.all)
### Step #10 - Walkthrough: Reading and writing image data: Read the TFRecord data
if args.step == 10: 
    print("\n### Step #10 - Walkthrough: Reading and writing image data: Read the TFRecord data")

    raw_image_dataset = tf.data.TFRecordDataset('tmp/tf2_t0305/images.tfrecords')

    # Create a dictionary describing the features.
    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_image_function(example_proto):
        # Parse the input tf.train.Example proto using the dictionary above.  
        return tf.io.parse_single_example(example_proto, image_feature_description)

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    logger.info(f'parsed_image_dataset:\n{parsed_image_dataset}')
        
    if args.plot:
        for image_features in parsed_image_dataset:
            image_raw = image_features['image_raw'].numpy()
            image = tf.image.decode_jpeg(image_raw)
            plt.figure()
            plt.imshow(image)
            plt.show(block=False)


### End of File
print()
if args.plot:
    plt.show()
debug()

