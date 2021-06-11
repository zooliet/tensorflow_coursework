#!/usr/bin/env python

# pip install -q tensorflow-hub
# pip install -q tensorflow-datasets

import sys
sys.path.append('./')
sys.path.append('../')

from lab_utils import (
    os, np, plt, logger, ap, BooleanAction,
    debug, toc, auto_increment
)

ap.add_argument('--epochs', type=int, default=10, help='number of epochs: 10*')
ap.add_argument('--batch', type=int, default=32, help='batch size: 32*')
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
import re
import shutil
import string

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("\n#################################################")
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense  

import tensorflow_hub as tfhub
import tensorflow_datasets as tfds


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - Download the IMDB dataset 
if args.step >= 1: 
    print("\n### Step #1 - Download the IMDB dataset ")

    # Split the training set into 60% and 40% to end up with 15,000 examples
    # for training, 10,000 examples for validation and 25,000 examples for testing.
    train_data, validation_data, test_data = tfds.load(
        name="imdb_reviews", 
        split=('train[:60%]', 'train[60%:]', 'test'),
        as_supervised=True
    )

args.step = auto_increment(args.step, args.all)
### Step #2 - Explore the IMDB dataset
if args.step == 2:
    print("\n### Step #2 - Explore the IMDB dataset")

    train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
    for i, (text, label) in enumerate(zip(train_examples_batch, train_labels_batch)):
        logger.info(f'{i+1}. {label.numpy()}: {text.numpy()[:30]}...')


args.step = auto_increment(args.step, args.all)
### Step #3 - Build the model
if args.step >= 3:
    print("\n### Step #3 - Build the model")

    # pre-trained text embedding model from TensorFlow Hub
    embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
    hub_layer = tfhub.KerasLayer(
        embedding, 
        input_shape=[],
        dtype=tf.string, 
        trainable=True
    )

    if args.step == 3:
        train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
        logger.info(hub_layer(train_examples_batch[:3]))

    model = Sequential()
    model.add(hub_layer)
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

    if args.step == 3:
        model.summary()

args.step = auto_increment(args.step, args.all)
### Step #4 - Build the model: Loss function and optimizer
if args.step >= 4:
    print("\n### Step #4 - Build the model: Loss function and optimizer")

    # Loss function and optimizer
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )


args.step = auto_increment(args.step, args.all)
### Step #5 - Train the model
if args.step >= 5:
    print("\n### Step #5 - Train the model")
    
    history = model.fit(
        train_data.shuffle(10000).batch(512),
        epochs=args.epochs,
        validation_data=validation_data.batch(512),
        verbose=2 if args.step == 5 else 0
    )

    if args.step == 5:
        for key, vals in history.history.items():
            logger.info(f'{key}: {list(map(lambda x: round(x,2), vals))}')


args.step = auto_increment(args.step, args.all)
### Step #6 - Evaludate the model 
if args.step == 6:
    print("\n### Step #6 - Evaluate the model")
    
    results = model.evaluate(test_data.batch(512), verbose=0)

    for name, value in zip(model.metrics_names, results):
        logger.info("%s: %.3f" % (name, value))


### End of File
if args.plot:
    plt.show()
debug()

