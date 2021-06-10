#!/usr/bin/env python

import sys
sys.path.append('./')
sys.path.append('../')

from lab_utils import (
    os, np, plt, logger, ap, BooleanAction,
    debug, toc
)

ap.add_argument('--epochs', type=int, default=10, help='number of epochs: 10*')
ap.add_argument('--batch', type=int, default=64, help='batch size: 64*')
args, extra_args = ap.parse_known_args()
logger.info(args)
# logger.info(extra_args)

if args.debug:
    import pdb
    import rlcompleter
    pdb.Pdb.complete=rlcompleter.Completer(locals()).complete
    # import code
    # code.interact(local=locals())
    debug = breakpoint

import time

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("\n#################################################")
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Concatenate


### Step #0 - TOC
if args.step == 0:
    toc(__file__)


### Step #1 - Load from .npz file
if args.step >= 1: 
    print("\n### Step #1 - Load from .npz file")

    DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

    path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
    with np.load(path) as data:
        train_examples = data['x_train']
        train_labels = data['y_train']
        test_examples = data['x_test']
        test_labels = data['y_test']

    if args.step == 1:
        logger.info(f'train_examples.shape: {train_examples.shape}')
        logger.info(f'test_examples.shape: {test_examples.shape}')
        logger.info(f'labels: {sorted(np.unique(train_labels))}')


### Step #2 - Load NumPy arrays with tf.data.Dataset
if args.step >= 2: 
    print("\n### Step #2 - Load NumPy arrays with tf.data.Dataset")

    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

    if args.step == 2:
        logger.info('train_dataset.element_spec:')
        print(*list(train_dataset.element_spec), sep='\n')

    
### Step #3 - Use the datasets: Shuffle and batch the datasets
if args.step >= 3: 
    print("\n### Step #3 - Use the datasets: Shuffle and batch the datasets")

    BATCH_SIZE = args.batch # 64
    SHUFFLE_BUFFER_SIZE = 100

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)


### Step #4 - Use the datasets: Build and train a model
if args.step == 4:
    print("\n### Step #4 - Use the datasets: Build and train a model")

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy']
    )

    model.fit(train_dataset, epochs=args.epochs, verbose=2)


### End of File
if args.plot:
    plt.show()
debug()

