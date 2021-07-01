#!/usr/bin/env python

# pip install -q pyyaml h5py

import sys
sys.path.append('./')
sys.path.append('../../')

from lab_utils import (
    tf, os, np, plt, logger, ap, BooleanAction,
    debug, toc, auto_increment
)

ap.add_argument('--epochs', type=int, default=3, help='number of epochs: 3*')
ap.add_argument('--batch', type=int, default=32, help='batch size: 32*')
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

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import tensorflow_datasets as tfds


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - Setup: Download the flowers dataset
if args.step >= 1: 
    print("\n### Step #1 - Setup: Download the flowers dataset")

    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    dataset_dir = tf.keras.utils.get_file(
        fname='flower_photos',
        origin=dataset_url,
        untar=True
    )
    # or
    # dataset_file = f'{os.getenv("HOME")}/.keras/datasets/flower_photos.tgz'
    # if not os.path.exists(dataset_file):
    #     dataset_file = tf.keras.utils.get_file('flower_photos.tgz', dataset_url, extract=True)
    # dataset_dir = os.path.join(os.path.dirname(dataset_file), 'flower_photos')
    
    dataset_dir = pathlib.Path(dataset_dir)
    image_count = len(list(dataset_dir.glob('*/*.jpg')))

    if args.step == 1:
        logger.info(f'dataset_dir: {dataset_dir}')
        logger.info(f'ls flower_photos/*/*.jpg | wc -l: {image_count}')

        if args.plot:
            roses = list(dataset_dir.glob('roses/*'))
            image = Image.open(str(roses[0]))
            plt.figure()
            plt.imshow(image)
            plt.show(block=False)
            # or
            # image = tf.io.read_file(str(roses[0]))
            # image = tf.image.decode_jpeg(image)
            # plt.figure()
            # plt.imshow(image)
            # plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #2 - Load using keras.preprocessing: Create a dataset
if args.step >= 2: 
    print("\n### Step #2 - Load using keras.preprocessing: Create a dataset")

    batch_size = args.batch # 32
    img_height = 180
    img_width = 180

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names

    if args.step == 2:
        print()
        logger.info(f'class names:\n{class_names}\n')
        for image_batch, labels_batch in train_ds.take(1):
            logger.info(f'image_batch.shape: {image_batch.shape}') # (32, 180, 180, 3)
            logger.info(f'labels_batch.shape: {labels_batch.shape}') # (32,)


args.step = auto_increment(args.step, args.all)
### Step #3 - Load using keras.preprocessing: Visualize the data
if args.step == 3: 
    print("\n### Step #3 - Load using keras.preprocessing: Visualize the data")    

    if args.plot == True:
        plt.figure(figsize=(10, 10))
        for images, labels in train_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(tf.cast(images[i], tf.uint8))
                # or plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #4 - Load using keras.preprocessing: Standardize the data
if args.step == 4: 
    print("\n### Step #4 - Load using keras.preprocessing: Standardize the data")

    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixels values are now in `[0,1]`.
    logger.info(f'normalization range: {np.min(first_image)}(min) ~ {np.max(first_image)}(max)')


args.step = auto_increment(args.step, args.all)
### Step #5 - Load using keras.preprocessing: Configure the dataset for performance
if args.step in [5, 6]: 
    print("\n### Step #5 - Load using keras.preprocessing: Configure the dataset for performance")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


args.step = auto_increment(args.step, args.all)
### Step #6 - Load using keras.preprocessing: Train a model
if args.step == 6: 
    print("\n### Step #6 - Load using keras.preprocessing: Train a model")

    num_classes = 5

    model = Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=2
    )


args.step = auto_increment(args.step, args.all)
### Step #7 - Using tf.data for finer control
if args.step >= 7: 
    print("\n### Step #7 - Using tf.data for finer control")

    list_ds = tf.data.Dataset.list_files(str(dataset_dir/'*/*'), shuffle=False)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

    class_names = sorted([item.name for item in dataset_dir.glob('*') if item.name != "LICENSE.txt"])

    if args.step == 7:
        logger.info(f'class_names:\n{class_names}\n')
        logger.info('list_ds:')
        for f in list_ds.take(5):
            print(f.numpy().decode())
        print("...\n")

    val_size = int(image_count * 0.2)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)

    # You can see the length of each dataset as follows:
    if args.step == 7:
        logger.info(f'# of tran_ds: {tf.data.experimental.cardinality(train_ds).numpy()}')
        logger.info(f'# of val_ds: {tf.data.experimental.cardinality(val_ds).numpy()}')

    # Write a short function that converts a file path to an (img, label) pair
    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-2] == class_names
        # Integer encode the label
        return tf.argmax(one_hot)

    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # resize the image to the desired size
        return tf.image.resize(img, [img_height, img_width])

    def process_path(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    if args.step == 7:
        for image, label in train_ds.take(1):
            logger.info("Image shape: {}".format(image.numpy().shape))
            logger.info("Label: {}".format(label.numpy()))


args.step = auto_increment(args.step, args.all)
### Step #8 - Using tf.data for finer control: Configure dataset for performance
if args.step >= 8: 
    print("\n### Step #8 - Using tf.data for finer control: Configure dataset for performance")

    def configure_for_performance(ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)


args.step = auto_increment(args.step, args.all)
### Step #9 - Using tf.data for finer control: Visualize the data
if args.step == 9: 
    print("\n### Step #9 - Using tf.data for finer control: Visualize the data")

    image_batch, label_batch = next(iter(train_ds))

    if args.plot:
        plt.figure(figsize=(10, 10))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(tf.cast(image_batch[i], tf.uint8))
            # plt.imshow(image_batch[i].numpy().astype("uint8"))
            label = label_batch[i]
            plt.title(class_names[label])
            plt.axis("off")
        plt.plot(block=False)
    else:
        logger.warning('use --plot for this step.')


args.step = auto_increment(args.step, args.all)
### Step #10 - Using tf.data for finer control: Continue training the model
if args.step == 10: 
    print("\n### Step #10 - Using tf.data for finer control: Continue training the model")

    num_classes = 5
    model = Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=2
    )

    
args.step = auto_increment(args.step, args.all)
### Step #11 - Using TensorFlow Datasets
if args.step == 11: 
    print("\n### Step #11 - Using TensorFlow Datasets")

    (train_ds, val_ds, test_ds), metadata = tfds.load(
        'tf_flowers',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True,
    )
    print()

    num_classes = metadata.features['label'].num_classes
    logger.info('# of classes: {}'.format(num_classes))

    if args.plot:
        get_label_name = metadata.features['label'].int2str
        image, label = next(iter(train_ds))
        plt.figure()
        _ = plt.imshow(image)
        _ = plt.title(get_label_name(label))
        plt.show(block=False)

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)
    test_ds = configure_for_performance(test_ds)


### End of File
print()
if args.plot:
    plt.show()
debug()

