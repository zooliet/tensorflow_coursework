#!/usr/bin/env python

import sys
sys.path.append('./')
sys.path.append('../')

from lab_utils import (
    os, np, plt, logger, ap, BooleanAction,
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
from PIL import Image
import pathlib

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("\n#################################################")
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Softmax
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import tensorflow_datasets as tfds


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - Download a dataset
if args.step >= 1:
    print("\n### Step #1 - Download a dataset")

    (train_ds, val_ds, test_ds), metadata = tfds.load(
        'tf_flowers',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True,
    )

    num_classes = metadata.features['label'].num_classes

    if args.step == 1:
        logger.info(f'num_classes: {num_classes}')
        if args.plot:
            image, label = next(iter(train_ds))
            get_label_name = metadata.features['label'].int2str
            plt.imshow(image)
            plt.title(get_label_name(label))
            plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #2 - Use Keras preprocessing layers: Resizing and rescaling
if args.step in [2, 3, 4, 5, 6, 7]:
    print("\n### Step #2 - Use Keras preprocessing layers: Resizing and rescaling")

    IMG_SIZE = 180

    resize_and_rescale = Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    ])

    image, label = next(iter(train_ds))
    result = resize_and_rescale(image)
    
    if args.step == 2:
        logger.info(f"result.shape: {result.shape}")
        logger.info(f"Min and max pixel values: {result.numpy().min()}, {result.numpy().max()}")

        if args.plot:
            _ = plt.imshow(result)
            plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #3 - Use Keras preprocessing layers: Data augmentation
if args.step in [3, 4, 5, 6, 7]:
    print("\n### Step #3 - Use Keras preprocessing layers: Data augmentation")

    data_augmentation = Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])
  
    # Add the image to a batch
    image = tf.expand_dims(image, 0)

    if args.step == 3 and args.plot:
        plt.figure(figsize=(10, 10))
        for i in range(9):
            augmented_image = data_augmentation(image)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_image[0])
            plt.axis("off")
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #4 - Use Keras preprocessing layers: Two options to use the preprocessing layers
if args.step == 4:
    print("\n### Step #4 - Use Keras preprocessing layers: Two options to use the preprocessing layers")

    # Option 1: Make the preprocessing layers part of your model
    model = Sequential([
        resize_and_rescale,
        data_augmentation,
        Conv2D(16, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        # Rest of your model
    ])

    # Option 2: Apply the preprocessing layers to your dataset
    aug_ds = train_ds.map(
        lambda x, y: (resize_and_rescale(x, training=True), y))


args.step = auto_increment(args.step, args.all)
### Step #5 - Use Keras preprocessing layers: Apply the preprocessing layers to the datasets
if args.step in [5, 6]:
    print("\n### Step #5 - Use Keras preprocessing layers: Apply the preprocessing layers to the datasets")

    batch_size = 32
    AUTOTUNE = tf.data.AUTOTUNE

    def prepare(ds, shuffle=False, augment=False):
        # Resize and rescale all datasets
        ds = ds.map(lambda x, y: 
            (resize_and_rescale(x), y), num_parallel_calls=AUTOTUNE)

        if shuffle:
            ds = ds.shuffle(1000)

        # Batch all datasets
        ds = ds.batch(batch_size)

        # Use data augmentation only on the training set
        if augment:
            ds = ds.map(lambda x, y: 
                (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

        # Use buffered prefecting on all datasets
        return ds.prefetch(buffer_size=AUTOTUNE)

    train_ds = prepare(train_ds, shuffle=True, augment=True)
    val_ds = prepare(val_ds)
    test_ds = prepare(test_ds)


args.step = auto_increment(args.step, args.all)
### Step #6 - Use Keras preprocessing layers: Train a model
if args.step == 6:
    print("\n### Step #6 - Use Keras preprocessing layers: Train a model")

    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    epochs=5
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs,
      verbose=2 if args.step == 6 else 0
    )

    loss, acc = model.evaluate(test_ds, verbose=0)
    if args.step == 6:
        logger.info("Accuracy: {:.4f}".format(acc))


args.step = auto_increment(args.step, args.all)
### Step #7 - Use Keras preprocessing layers: Custom data augmentation
if args.step == 7:
    print("\n### Step #7 - Use Keras preprocessing layers: Custom data augmentation")

    def random_invert_img(x, p=0.5):
        if  tf.random.uniform([]) < p:
            x = (255-x)
        else:
            x
        return x

    def random_invert(factor=0.5):
        return tf.keras.layers.Lambda(lambda x: random_invert_img(x, factor))

    random_invert = random_invert()

    if args.plot:
        plt.figure(figsize=(10, 10))
        for i in range(9):
            augmented_image = random_invert(image)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_image[0].numpy().astype("uint8"))
            plt.axis("off")
        plt.show(block=False)

    class RandomInvert(tf.keras.layers.Layer):
        def __init__(self, factor=0.5, **kwargs):
            super().__init__(**kwargs)
            self.factor = factor

        def call(self, x):
            return random_invert_img(x)

    if args.plot:
        plt.figure(figsize=(10, 10))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(RandomInvert()(image)[0])
            plt.axis("off")
        plt.show(block=False)
        

args.step = auto_increment(args.step, args.all)
### Step #8 - Using tf.image
if args.step >= 8:
    print("\n### Step #8 - Using tf.image")

    (train_ds, val_ds, test_ds), metadata = tfds.load(
        'tf_flowers',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True,
    )
    
    num_classes = metadata.features['label'].num_classes

    image, label = next(iter(train_ds))
    get_label_name = metadata.features['label'].int2str
    
    if args.step == 8:
        logger.info(f'num_classes: {num_classes}')
        if args.plot:
            plt.figure()
            plt.imshow(image)
            plt.title(get_label_name(label))
            plt.show(block=False)

    def visualize(original, augmented):
        fig = plt.figure()
        plt.subplot(1,2,1)
        plt.title('Original image')
        plt.imshow(original)

        plt.subplot(1,2,2)
        plt.title('Augmented image')
        plt.imshow(augmented)


args.step = auto_increment(args.step, args.all)
### Step #9 - Using tf.image: Data augmentation
if args.step == 9:
    print("\n### Step #9 - Using tf.image: Data augmentation")

    flipped = tf.image.flip_left_right(image)
    grayscaled = tf.image.rgb_to_grayscale(image)
    saturated = tf.image.adjust_saturation(image, 3)
    bright = tf.image.adjust_brightness(image, 0.4)
    cropped = tf.image.central_crop(image, central_fraction=0.5)
    rotated = tf.image.rot90(image)

    if args.plot:
        visualize(image, flipped)
        visualize(image, tf.squeeze(grayscaled))
        _ = plt.colorbar()
        visualize(image, saturated)
        visualize(image, bright)
        visualize(image,cropped)
        visualize(image, rotated)


args.step = auto_increment(args.step, args.all)
### Step #10 - Using tf.image: Random transformations
if args.step == 10:
    print("\n### Step #10 - Using tf.image: Random transformations")

    if args.plot:
        for i in range(3):
            seed = (i, 0)  # tuple of size (2,)
            stateless_random_brightness = tf.image.stateless_random_brightness(
                image, max_delta=0.95, seed=seed)
            visualize(image, stateless_random_brightness)

        for i in range(3):
            seed = (i, 0)  # tuple of size (2,)
            stateless_random_contrast = tf.image.stateless_random_contrast(
                image, lower=0.1, upper=0.9, seed=seed)
            visualize(image, stateless_random_contrast)

        for i in range(3):
            seed = (i, 0)  # tuple of size (2,)
            stateless_random_crop = tf.image.stateless_random_crop(
                image, size=[210, 300, 3], seed=seed)
            visualize(image, stateless_random_crop)


args.step = auto_increment(args.step, args.all)
### Step #11 - Using tf.image: Apply augmentation to a dataset
if args.step == 11:
    print("\n### Step #11 - Using tf.image: Apply augmentation to a dataset")
    
    AUTOTUNE = tf.data.AUTOTUNE
    IMG_SIZE = 180 
    batch_size = 32

    (train_datasets, val_ds, test_ds), metadata = tfds.load(
        'tf_flowers',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True,
    )

    def resize_and_rescale(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = (image / 255.0)
        return image, label

    def augment(image_label, seed):
        image, label = image_label
        image, label = resize_and_rescale(image, label)
        image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
        # Make a new seed
        new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
        # Random crop back to the original size
        image = tf.image.stateless_random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
        # Random brightness
        image = tf.image.stateless_random_brightness(image, max_delta=0.5, seed=new_seed)
        image = tf.clip_by_value(image, 0, 1)
        return image, label

    # Option 1: Using tf.data.experimental.Counter()

    # Create counter and zip together with train dataset
    counter = tf.data.experimental.Counter()
    train_ds = tf.data.Dataset.zip((train_datasets, (counter, counter)))

    train_ds = (
        train_ds
            .shuffle(1000)
            .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        val_ds
            .map(resize_and_rescale, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        test_ds
            .map(resize_and_rescale, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
    )


    # Option 2: Using tf.random.Generator

    # Create a generator
    rng = tf.random.Generator.from_seed(123, alg='philox')

    # A wrapper function for updating seeds
    def f(x, y):
        seed = rng.make_seeds(2)[0]
        image, label = augment((x, y), seed)
        return image, label

    train_ds = (
        train_datasets
            .shuffle(1000)
            .map(f, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE)
    )

    val_ds = (
        val_ds
            .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE)
    )

    test_ds = (
        test_ds
            .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE)
    )


### End of File
if args.plot:
    plt.show()
debug()


