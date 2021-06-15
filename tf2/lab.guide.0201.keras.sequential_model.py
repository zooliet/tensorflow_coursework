#!/usr/bin/env python

import sys
sys.path.append('./')
sys.path.append('../')

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
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - When to use a Sequential model
if args.step == 1:
    print("\n### Step #1 - When to use a Sequential model")

    logger.info('A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.')

    # Define Sequential model with 3 layers
    model = Sequential([
        Dense(2, activation="relu", name="layer1"),
        Dense(3, activation="relu", name="layer2"),
        Dense(4, name="layer3"),
    ])

    # Call model on a test input
    x = tf.ones((3, 3))
    y = model(x)

    logger.info('The model above is equivalent to this function')    

    # Create 3 layers
    layer1 = Dense(2, activation="relu", name="layer1")
    layer2 = Dense(3, activation="relu", name="layer2")
    layer3 = Dense(4, name="layer3")

    # Call layers on a test input
    x = tf.ones((3, 3))
    y = layer3(layer2(layer1(x)))


args.step = auto_increment(args.step, args.all)
### Step #2 - Creating a Sequential model
if args.step == 2:
    print("\n### Step #2 - Creating a Sequential model")
    
    model = Sequential([
        Dense(2, activation="relu"),
        Dense(3, activation="relu"),
        Dense(4),
    ])
    logger.info('model.layers:')
    print(*[layer.name for layer in model.layers], sep='\n')
    print('')

    model = Sequential()
    model.add(Dense(2, activation="relu"))
    model.add(Dense(3, activation="relu"))
    model.add(Dense(4))
    logger.info('model.layers:')
    print(*[layer.name for layer in model.layers], sep='\n')
    print('')

    model.pop()
    logger.info('after model.pop():')
    print(*[layer.name for layer in model.layers], sep='\n')
    print('')
    logger.info("len(model.layers): {}".format(len(model.layers))) # 2

    model = Sequential(name="my_sequential")
    model.add(Dense(2, activation="relu", name="layer1"))
    model.add(Dense(3, activation="relu", name="layer2"))
    model.add(Dense(4, name="layer3"))
    logger.info('model.layers:')
    print(*[layer.name for layer in model.layers], sep='\n')


args.step = auto_increment(args.step, args.all)
### Step #3 - Specifying the input shape in advance
if args.step == 3:
    print("\n### Step #3 - Specifying the input shape in advance")

    layer = Dense(3)
    logger.info(f'layer.weights: {layer.weights}')  # Empty

    x = tf.ones((1, 4))
    y = layer(x)
    # Now it has weights, of shape (4, 3) and (3,) 
    logger.info('layer.weights')
    print(*[f"{weight.name}:{weight.shape}" for weight in layer.weights], sep='\n')
    print('')


    model = Sequential([
        Dense(2, activation="relu"),
        Dense(3, activation="relu"),
        Dense(4),
    ])  
    # No weights at this stage!
    # At this point, you can't do this:  model.weights
    # You also can't do this:  model.summary()

    # Call the model on a test input
    x = tf.ones((1, 4))
    y = model(x)

    logger.info('layer.weights')
    print(*[f"{weight.name}:{weight.shape}" for weight in layer.weights], sep='\n')
    print('')

    # Once a model is "built", you can call its summary() method to display its contents:
    model.summary()


args.step = auto_increment(args.step, args.all)
### Step #4 -- Specifying the input shape in advance
if args.step == 4:
    print("\n### Step #3 -- Specifying the input shape in advance")

    # it can be very useful if you start your model by passing an Input object to your model, so that it knows its input shape from the start:
    model = Sequential()
    # model.add(Input(shape=(4,))) # old way
    model.add(InputLayer(input_shape=(4,)))
    model.add(Dense(2, activation="relu"))
    model.summary()

    # or
    model = Sequential()
    model.add(Dense(2, activation="relu", input_shape=(4,)))
    model.summary()


args.step = auto_increment(args.step, args.all)
### Step #5 - A common debugging workflow: add() + summary()
if args.step == 5:
    print("\n### Step #4 - A common debugging workflow: add() + summary()")

    model = Sequential()
    # model.add(Input(shape=(250, 250, 3)))  # 250x250 RGB images
    model.add(InputLayer(input_shape=(250, 250, 3)))  # 250x250 RGB images
    model.add(Conv2D(32, 5, strides=2, activation="relu"))
    model.add(Conv2D(32, 3, activation="relu"))
    model.add(MaxPooling2D(3))

    # Can you guess what the current output shape is at this point? Probably not.
    # Let's just print it:
    model.summary()

    # The answer was: (40, 40, 32), so we can keep downsampling...
    model.add(Conv2D(32, 3, activation="relu"))
    model.add(Conv2D(32, 3, activation="relu"))
    model.add(MaxPooling2D(3))
    model.add(Conv2D(32, 3, activation="relu"))
    model.add(Conv2D(32, 3, activation="relu"))
    model.add(MaxPooling2D(2))

    # And now?
    model.summary()

    # Now that we have 4x4 feature maps, time to apply global max pooling.
    model.add(GlobalMaxPooling2D())

    # Finally, we add a classification layer.
    model.add(Dense(10))


args.step = auto_increment(args.step, args.all)
### Step #6 - Feature extraction with a Sequential model
if args.step == 6:
    print("\n### Step #6 - Feature extraction with a Sequential model")

    initial_model = Sequential( [
        # Input(shape=(250, 250, 3)),
        InputLayer(input_shape=(250, 250, 3)),
        Conv2D(32, 5, strides=2, activation="relu"),
        Conv2D(32, 3, activation="relu"),
        Conv2D(32, 3, activation="relu"),
    ])

    logger.info('model.layers:')
    print(*[layer.output.name for layer in initial_model.layers], sep='\n')
    print('')

    feature_extractor = Model(
        inputs=initial_model.inputs,
        outputs=[layer.output for layer in initial_model.layers],
    )

    # Call feature extractor on test input.
    x = tf.ones((1, 250, 250, 3))
    features = feature_extractor(x)
    logger.info(f'len(features): {len(features)}')
    logger.info('shape of each feature:')
    print(*[feature.shape for feature in features], sep='\n')
    print('')

    # Here's a similar example that only extract features from one layer:
    initial_model = Sequential( [
        InputLayer(input_shape=(250, 250, 3)),
        Conv2D(32, 5, strides=2, activation="relu"),
        Conv2D(32, 3, activation="relu", name="my_intermediate_layer"),
        Conv2D(32, 3, activation="relu"),
    ])

    feature_extractor = Model(
        inputs=initial_model.inputs,
        outputs=initial_model.get_layer(name="my_intermediate_layer").output,
    )

    # Call feature extractor on test input.
    x = tf.ones((1, 250, 250, 3))
    features = feature_extractor(x)
    logger.info(f'len(features): {len(features)}')
    logger.info('shape of each feature:')
    print(*[feature.shape for feature in features], sep='\n')
    print('')


args.step = auto_increment(args.step, args.all)
### Step #7 - Transfer learning with a Sequential model
if args.step == 7:
    print("\n### Step #7 - Transfer learning with a Sequential model")

    model = Sequential([
        InputLayer(input_shape=(784)),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(10),
    ])

    # Presumably you would want to first load pre-trained weights.
    # model.load_weights(...)

    # Freeze all layers except the last one.
    for layer in model.layers[:-1]:
        layer.trainable = False

    # Recompile and train (this will only update the weights of the last layer).
    # model.compile(...)
    # model.fit(...)

    # Another common blueprint is to use a Sequential model to stack a pre-trained model

    # Load a convolutional base with pre-trained weights
    base_model = tf.keras.applications.Xception(
        weights='imagenet',
        include_top=False,
        pooling='avg')

    # Freeze the base model
    base_model.trainable = False

    # Use a Sequential model to add a trainable classifier on top
    model = Sequential([
        base_model,
        Dense(1000),
    ])

    # Compile & train
    # model.compile(...)
    # model.fit(...)


### End of File
if args.plot:
    plt.show()
debug()
