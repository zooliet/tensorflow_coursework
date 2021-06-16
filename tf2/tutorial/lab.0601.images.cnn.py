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

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Softmax
from tensorflow.keras.layers import Conv2D, MaxPooling2D


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - Download and prepare the CIFAR10 dataset
if args.step >= 1:
    print("\n### Step #1 - Download and prepare the CIFAR10 dataset")

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0


args.step = auto_increment(args.step, args.all)
### Step #2 - Verify the data
if args.step >= 2:
    print("\n### Step #2 - Verify the data")

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    if args.step == 2 and args.plot:
        plt.figure(figsize=(10,10))
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i])
            # The CIFAR labels happen to be arrays, 
            # which is why you need the extra index
            plt.xlabel(class_names[train_labels[i][0]])
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #3 - Create the convolutional base
if args.step >= 3:
    print("\n### Step #3 - Create the convolutional base")

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    if args.step == 3:
        model.summary()


args.step = auto_increment(args.step, args.all)
### Step #4 - Add Dense layers on top
if args.step >= 4:
    print("\n### Step #4 - Add Dense layers on top")

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10))

    if args.step == 4:
        model.summary()


args.step = auto_increment(args.step, args.all)
### Step #5 - Compile and train the model
if args.step >= 5:
    print("\n### Step #5 - Compile and train the model")

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = model.fit(
        train_images, 
        train_labels, 
        epochs=args.epochs,
        validation_data=(test_images, test_labels),
        verbose=2 if args.step == 5 else 0
    )


args.step = auto_increment(args.step, args.all)
### Step #6 - Evaluate the model
if args.step == 6:
    print("\n### Step #6 - Evaluate the model")

    if args.step == 6 and args.plot:
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show(block=False)

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=0)
    logger.info('test_loss: {:.2f}'.format(test_loss))
    logger.info('test_acc: {:4.1f}%'.format(test_acc*100))


### End of File
if args.plot:
    plt.show()
debug()


