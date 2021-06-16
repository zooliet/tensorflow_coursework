#!/usr/bin/env python

import sys
sys.path.append('./')
sys.path.append('../../')

from lab_utils import (
    tf, os, np, plt, logger, ap, BooleanAction,
    debug, toc, auto_increment
)

ap.add_argument('--epochs', type=int, default=10, help='number of epochs: 10*')
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

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Softmax
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - Import the MNIST dataset
if args.step:
    print("\n### Step #1 - Import the MNIST dataset")

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0

    x_train = x_train[..., tf.newaxis].astype("float32") # (60000,28,28) =>(60000,28,28,1)
    x_test = x_test[..., tf.newaxis].astype("float32") # (10000,28,28) =>(10000,28,28,1)
    # x_train, x_test = tf.cast(x_train, tf.float32), tf_cast(x_test, tf.float32)
    # y_train, y_test = tf.cast(y_train, tf.int64), tf.cast(y_test, tf.int64)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(10000).batch(32)
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


args.step = auto_increment(args.step, args.all)
### Step #2 - Prepare model templates
if args.step >= 2:
    print("\n### Step #2 - Prepare model templates")

    def get_sequential_model():
        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(None, None, 1)),
            Conv2D(16, (3,3), activation='relu'),
            GlobalAveragePooling2D(),
            Dense(10)
        ]) 
        return model

    def get_functional_model():
        inputs = Input(shape=(28,28,1))
        x = Conv2D(32, (3,3), activation='relu')(inputs)
        x = Conv2D(16, (3,3), activation='relu')(x)
        x = GlobalAveragePooling2D()(x)
        outputs = Dense(10)(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model


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
    logger.info(f"loss before training: {batch_loss:.2f}")

    epochs = args.epochs # 10
    start = time.time()
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, shuffle=True, verbose=2)
    end = time.time()
    logger.info(f"Sequential model with fit(x_train,y_train): {end - start:.2f} secs\n")

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    logger.info(f'Test accuracy: {test_acc:.4f}')
    
    predictions = model(x_test[:3])
    logger.info(f'Predictions(logits):\n{predictions}')
    logger.info(f'Predictions(softmax):\n{tf.nn.softmax(predictions)}')
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
            tf.debugging.assert_equal(predictions.shape, (32,10))
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
    logger.info(f"Sequential model with custom training loop: {end - start:.2f} secs\n")

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = loss_fn(labels, predictions)

        test_loss.update_state(t_loss)
        test_accuracy.update_state(labels, predictions)

    test_loss.reset_states()
    test_accuracy.reset_states()

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    logger.info(f'Test accuracy: {test_accuracy.result():.4f}')


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
    end  = time.time()
    logger.info(f"Functional model with fit(x_train,y_train): {end - start:.2f} secs\n")


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
    logger.info(f"Functional model with fit(train_ds)")

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    logger.info(f'Test accuracy: {test_acc:.4f}')


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
            tf.debugging.assert_equal(predictions.shape, (32,10))
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
    logger.info(f"Functional model with custom train loop: {end - start:.2f} secs\n")

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = loss_fn(labels, predictions)

        test_loss.update_state(t_loss)
        test_accuracy.update_state(labels, predictions)

    test_loss.reset_states()
    test_accuracy.reset_states()

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    logger.info(f'Test accuracy: {test_accuracy.result():.4f}')


### End of File
if args.plot:
    plt.show()
debug()
