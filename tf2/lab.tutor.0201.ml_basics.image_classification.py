#!/usr/bin/env python

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

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("\n#################################################")
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Softmax
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - Import the Fashion MNIST dataset
if args.step >= 1:
    print("\n### Step #1 - Import the Fashion MNIST dataset")

    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]


args.step = auto_increment(args.step, args.all)
### Step #2 - Explore the data
if args.step == 2:
    print("\n### Step #2 - Explore the data")

    logger.info(f'train_images.shape: {train_images.shape}') # (60000, 28, 28)
    logger.info(f'len(train_labels): {len(train_labels)}') # 60000
    logger.info(f'test_images.shape: {test_images.shape}') # (10000, 28, 28)
    logger.info(f'len(test_labels): {len(test_labels)}') # 10000

    logger.info(f'train_labels: {np.unique(train_labels)}')


args.step = auto_increment(args.step, args.all)
### Step #3 - Preprocess the data
if args.step >= 3:
    print("\n### Step #3 - Preprocess the data")

    if args.step == 3 and args.plot:
        plt.figure()
        plt.imshow(train_images[0])
        plt.colorbar()
        plt.grid(False)
        plt.show(block=False)

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    if args.step == 3 and args.plot:
        plt.figure(figsize=(10,10))
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i], cmap=plt.cm.binary)
            plt.xlabel(class_names[train_labels[i]])
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #4 - Build the model: Set up the layers
if args.step >= 4:
    print("\n### Step #4 - Build the model: Set up the layers")

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10)
    ])


args.step = auto_increment(args.step, args.all)
### Step #5 - Build the model: Compile the model
if args.step >= 5:
    print("\n### Step #5 - Build the model: Compile the model")

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )


args.step = auto_increment(args.step, args.all)
### Step #6 - Train the model: Feed the model
if args.step >= 6:
    print("\n### Step #6 - Train the model: Feed the model")

    model.fit(
        train_images, train_labels, 
        epochs=args.epochs, 
        verbose=2 if args.step == 6 else 0
    )


args.step = auto_increment(args.step, args.all)
### Step #7 - Train the model: Evaluate accuracy
if args.step == 7:
    print("\n### Step #7 - Train the model: Evaluate accuracy")

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=0)
    logger.info(f'Test accuracy: {test_acc:.4f}')


args.step = auto_increment(args.step, args.all)
### Step #8 - Train the model: Make predictions
if args.step >= 8:
    print("\n### Step #8 - Train the model: Make predictions")

    probability_model = Sequential([model, Softmax()])
    predictions = probability_model.predict(test_images)
    if args.step == 8:
        logger.info(f'Predictions(softmax):\n{predictions[:5]}')
        logger.info(f'After applying np.argmax: {np.argmax(predictions[:5], -1)}')

    def plot_image(i, predictions_array, true_label, img):
        true_label, img = true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'
        plt.xlabel("{} {:2.0f}% ({})".format(
            class_names[predicted_label],
            100*np.max(predictions_array),
            class_names[true_label]),
            color=color)

    def plot_value_array(i, predictions_array, true_label):
        true_label = true_label[i]
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')


args.step = auto_increment(args.step, args.all)
### Step #9 - Train the model: Verify predictions
if args.step == 9: 
    print("\n### Step #9 - Train the model: Verify predictions")

    if args.plot:
        i = 0
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(1,2,2)
        plot_value_array(i, predictions[i],  test_labels)
        plt.show(block=False)

        i = 12
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(1,2,2)
        plot_value_array(i, predictions[i],  test_labels)
        plt.show(block=False)
    
        # Plot the first X test images, their predicted labels, and the true labels.
        # Color correct predictions in blue and incorrect predictions in red.
        num_rows = 5
        num_cols = 3
        num_images = num_rows*num_cols

        plt.figure(figsize=(2*2*num_cols, 2*num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2*num_cols, 2*i+1)
            plot_image(i, predictions[i], test_labels, test_images)
            plt.subplot(num_rows, 2*num_cols, 2*i+2)
            plot_value_array(i, predictions[i], test_labels)
        plt.tight_layout()
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #10 - Use the trained model
if args.step == 10: 
    print("\n### Step #10 - Use the trained model")

    # with single test image
    img = test_images[1]
    logger.info(f'test_image.shape: {img.shape}')

    # Add the image to a batch where it's the only member.
    img = np.expand_dims(img, 0) # (28, 28) => (1, 28, 28)
    logger.info(f'test_image.shape: {img.shape}')

    predictions_single = probability_model.predict(img)
    logger.info(f'predictions: {predictions_single}')
    logger.info(f'np.argmax(): {np.argmax(predictions_single[0])}')

    if args.plot:
        plt.figure(figsize=(6,3))
        plot_value_array(1, predictions_single[0], test_labels)
        _ = plt.xticks(range(10), class_names, rotation=45)
        plt.show(block=False)


### End of File
if args.plot:
    plt.show()
debug()

