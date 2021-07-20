#!/usr/bin/env python

# pip install -U tfx

import sys
sys.path.append('./')
sys.path.append('../../')

from lab_utils import (
    tf, os, np, plt, logger, ap, BooleanAction,
    debug, toc, auto_increment
)

# ap.add_argument('--epochs', type=int, default=10, help='number of epochs: 10*')
# ap.add_argument('--batch', type=int, default=32, help='batch size: 32*')
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
import urllib.request
import tempfile
import pandas as pd

# from tfx import v1 as tfx
import tfx
print('TFX version: {}'.format(tfx.__version__))

# from tensorflow.keras import Sequential, Model, Input
# from tensorflow.keras.layers import Flatten, Dense, Dropout, Softmax


### TOC
if args.step == 0:
    toc(__file__)


if args.step or args.all:
    if not os.path.exists('tmp/tfx_t0101/'):
        os.mkdir('tmp/tfx_t0101/') 


args.step = auto_increment(args.step, args.all)
### Step #1 - Setup: Set up variables
if args.step >= 1:
    print("\n### Step #1 - Setup: Set up variables")

    PIPELINE_NAME = "penguin-simple"

    # Output directory to store artifacts generated from the pipeline.
    PIPELINE_ROOT = os.path.join('tmp/tfx_t0101/pipelines', PIPELINE_NAME)
    # Path to a SQLite DB file to use as an MLMD storage.
    METADATA_PATH = os.path.join('tmp/tfx_t0101/metadata', PIPELINE_NAME, 'metadata.db')
    # Output directory where created models from the pipeline will be exported.
    SERVING_MODEL_DIR = os.path.join('tmp/tfx_t0101/serving_model', PIPELINE_NAME)

    # from absl import logging
    # logging.set_verbosity(logging.INFO)  # Set default logging level.



args.step = auto_increment(args.step, args.all)
### Step #2 - Setup: Prepare example data
if args.step >= 2:
    print("\n### Step #2 - Setup: Prepare example data")

    _data_url = 'https://raw.githubusercontent.com/tensorflow/tfx/master/tfx/examples/penguin/data/labelled/penguins_processed.csv'
    DATA_ROOT = tf.keras.utils.get_file(
        fname='data.csv',
        origin=_data_url, 
        cache_subdir='datasets/tfx-data'
    )

    if args.step == 2:
        logger.info(f'DATA_ROOT: {DATA_ROOT}')
        df = pd.read_csv(DATA_ROOT)
        print(df.head())





args.step = auto_increment(args.step, args.all)
### Step #3 - Build a sequentail model, and fit() with (x_train,y_train)
if args.step == 3:
    print("\n### Step #3 - Build a sequentail model, and fit() with (x_train,y_train)")

    model = get_sequential_model()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(
        optimizer = optimizer,
        loss = loss_fn,
        metrics = ['accuracy']
    )
    model.summary()

    features, labels = next(iter(train_ds))
    predictions = model(features) 
    batch_loss = loss_fn(labels, predictions)
    logger.info(f"loss before training: {batch_loss:.2f}\n")

    epochs = args.epochs # 10
    start = time.time()
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, shuffle=True, verbose=2)
    end = time.time()
    print()
    
    logger.info(f"Sequential model with fit(x_train,y_train): {end - start:.2f} secs\n")

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    logger.info(f'Test accuracy: {test_acc:.4f}\n')
    
    predictions = model(x_test[:3])
    logger.info(f'Predictions(logits):\n{predictions}\n')
    logger.info(f'Predictions(softmax):\n{tf.nn.softmax(predictions)}\n')
    # or
    probability_model = Sequential([model, Softmax()])
    predictions = probability_model(x_test[:3])
    logger.info(f'Predictions(softmax-integrated):\n{predictions}')


args.step = auto_increment(args.step, args.all)
### Step #4 - Build a sequentail model, and fit() with train_ds
if args.step == 4:
    print("\n### Step #4 - Build a sequentail model, and fit() with train_ds")

    model = get_sequential_model()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(
        optimizer = optimizer,
        loss = loss_fn,
        metrics = ['accuracy']
    )

    epochs = args.epochs # 10
    start = time.time()
    model.fit(train_ds, epochs=epochs, verbose=2)
    end = time.time()
    print()
    logger.info(f"Sequential model with fit(train_ds): {end - start:.2f} secs\n")

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    logger.info(f'Test accuracy: {test_acc:.4f}')


args.step = auto_increment(args.step, args.all)
### Step #5 - Build a sequentail model, and train with custom loop
if args.step == 5:
    print("\n### Step #5 - Build a sequentail model, and train with custom loop")

    model = get_sequential_model()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss.update_state(loss)
        train_accuracy.update_state(labels, predictions)

    start = time.time()
    epochs = args.epochs # 10
    for epoch in range(epochs):
        for batch, (images, labels) in enumerate(train_ds):
            train_step(images, labels)

        # logger.info(
        #     f'Epoch {epoch+1}/{epochs} => loss: {train_loss.result():.4f} - accuracy: {train_accuracy.result():.4f}'
        # )

        # Display metrics at the end of each epoch.
        t_loss = train_loss.result()
        t_acc = train_accuracy.result()
        logger.info(f"Epoch {epoch}/{epochs} - loss: {t_loss:.4f} - accuracy: {t_acc:.4f}")

        # Reset training metrics at the end of each epoch
        train_loss.reset_states()
        train_accuracy.reset_states()

    end = time.time()
    logger.info(f"Sequential model with custom train loop: {end - start:.2f} secs")


args.step = auto_increment(args.step, args.all)
### Step #6 - Build a functional model, and fit() with (x_train,y_train) 
if args.step == 6:
    print("\n### Step #6 - Build a functional model, and fit() with (x_train,y_train)")

    model = get_functional_model()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(
        optimizer = optimizer,
        loss = loss_fn,
        metrics = ['accuracy']
    )

    epochs = args.epochs # 10
    start = time.time()
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, shuffle=True, verbose=2)
    end = time.time()
    print()
    logger.info(f"Functional model with fit(x_train,y_train): {end - start:.2f} secs")


args.step = auto_increment(args.step, args.all)
### Step #7 - Build a functional model, and fit() with train_ds
if args.step == 7:
    print("\n### Step #7 - Build a functional model, and fit() with train_ds")

    model = get_functional_model()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(
        optimizer = optimizer,
        loss = loss_fn,
        metrics = ['accuracy']
    )

    epochs = args.epochs # 10
    start = time.time()
    model.fit(train_ds, epochs=epochs, verbose=2)
    end = time.time()
    print()
    logger.info(f"Functional model with fit(drain_ds): {end - start:.2f} secs")


args.step = auto_increment(args.step, args.all)
### Step #8 - Build a functional model, and train with custom loop
if args.step == 8:
    print("\n### Step #8 - Build a functional model, and train with custom loop")

    model = get_functional_model()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # train_loss(loss)
        # train_accuracy(labels, predictions)
        train_loss.update_state(loss)
        train_accuracy.update_state(labels, predictions)

    start = time.time()
    epochs = args.epochs # 10
    for epoch in range(epochs):
        for batch, (images, labels) in enumerate(train_ds):
            train_step(images, labels)

        # logger.info(
        #     f'Epoch {epoch+1}/{epochs} => loss: {train_loss.result():.4f} - accuracy: {train_accuracy.result():.4f}'
        # )

        # Display metrics at the end of each epoch.
        t_loss = train_loss.result()
        t_acc = train_accuracy.result()
        logger.info(f"Epoch {epoch}/{epochs} - loss: {t_loss:.4f} - accuracy: {t_acc:.4f}")

        # Reset training metrics at the end of each epoch
        train_loss.reset_states()
        train_accuracy.reset_states()

    end = time.time()
    logger.info(f"Functional model with custom train loop: {end - start:.2f} secs")
   

### End of File
print()
if args.plot:
    plt.show()
debug()
