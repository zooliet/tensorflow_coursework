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
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - When to use a Sequential model
if args.step == 1:
    print("\n### Step #1 - When to use a Sequential model")

    __doc__='''
    A Sequential model is appropriate for a plain stack of layers where each
    layer has exactly one input tensor and one output tensor.

    A Sequential model is not appropriate when:
    - Your model has multiple inputs or multiple outputs
    - Any of your layers has multiple inputs or multiple outputs
    - You need to do layer sharing
    - You want non-linear topology (e.g. a residual connection, a multi-branch
      model)
    '''
    print(__doc__)

    # Define Sequential model with 3 layers
    model = Sequential([
        Dense(2, activation="relu", name="layer1"),
        Dense(3, activation="relu", name="layer2"),
        Dense(4, name="layer3"),
    ])

    # Call model on a test input
    x = tf.ones((3, 3))
    y = model(x)

    # The model above is equivalent to this function 
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
    
    logger.info('create a Sequential model by passing a list of layers to the Sequential constructor:')
    model = Sequential([
        Dense(2, activation="relu"),
        Dense(3, activation="relu"),
        Dense(4),
    ])
    logger.info('model.layers:')
    print(*[layer.name for layer in model.layers], sep='\n')
    print('')

    logger.info('create a Sequential model incrementally via the add() method:')
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
    print('')


args.step = auto_increment(args.step, args.all)
### Step #3 - Specifying the input shape in advance
if args.step == 3:
    print("\n### Step #3 - Specifying the input shape in advance")

    __doc__='''
    Generally, all layers in Keras need to know the shape of their inputs in
    order to be able to create their weights. So when you create a layer like
    this, initially, it has no weights.

    It creates its weights the first time it is called on an input, since the
    shape of the weights depends on the shape of the inputs.
    '''
    print(__doc__)

    layer = Dense(3)
    logger.info(f'layer.weights: {layer.weights}')  # Empty

    x = tf.ones((1, 4))
    y = layer(x)
    # Now it has weights, of shape (4, 3) and (3,) 
    logger.info('layer.weights:')
    print(*[f"{weight.name} {weight.shape}" for weight in layer.weights], sep='\n')
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

    logger.info('layer.weights:')
    print(*[f"{weight.name} {weight.shape}" for weight in layer.weights], sep='\n')

    __doc__='''
    Once a model is "built", you can call its summary() method to display its
    contents.

    However, it can be very useful when building a Sequential model
    incrementally to be able to display the summary of the model so far,
    including the current output shape.  In this case, you should start your
    model by passing an InputLayer object to your model, so that it knows its
    input shape from the start.

    Models built with a predefined input shape like this always have weights
    (even before seeing any data) and always have a defined output shape.  In
    general, it's a recommended best practice to always specify the input shape
    of a Sequential model in advance if you know what it is.
    '''
    print(__doc__)

    model = Sequential()
    model.add(InputLayer(input_shape=(4,)))
    model.add(Dense(2, activation="relu"))
    model.summary()
    print('')

    logger.info("InputLayer object is not displayed as part of model.layers:")
    print(*[layer.name for layer in model.layers], sep='\n')
    print('')

    logger.info('A simple alternative is to just pass an input_shape argument to your first layer:')
    model = Sequential()
    model.add(Dense(2, activation="relu", input_shape=(4,)))
    model.summary()
    print('')


args.step = auto_increment(args.step, args.all)
### Step #4 - A common debugging workflow: add() + summary()
if args.step == 4:
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
    print('')

    # Now that we have 4x4 feature maps, time to apply global max pooling.
    model.add(GlobalMaxPooling2D())

    # Finally, we add a classification layer.
    model.add(Dense(10))


args.step = auto_increment(args.step, args.all)
### Step #5 - What to do once you have a model
if args.step == 5:
    print("\n### Step #5 - What to do once you have a model")

    __doc__='''
    Once your model architecture is ready, you will want to:
    - Train your model, evaluate it, and run inference. See our guide to
      training & evaluation with the built-in loops
    - Save your model to disk and restore it. See our guide to serialization &
      saving.
    - Speed up model training by leveraging multiple GPUs. See our guide to
      multi-GPU and distributed training.
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #6 - Feature extraction with a Sequential model
if args.step == 6:
    print("\n### Step #6 - Feature extraction with a Sequential model")

    __doc__='''
    Once a Sequential model has been built, it behaves like a Functional API
    model. This means that every layer has an input and output attribute. These
    attributes can be used to do neat things, like quickly creating a model
    that extracts the outputs of all intermediate layers in a Sequential model.
    '''
    print(__doc__)

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
    
    __doc__='''
    Transfer learning consists of freezing the bottom layers in a model and
    only training the top layers. If you aren't familiar with it, make sure to
    read our guide to transfer learning.

    Here are two common transfer learning blueprint involving Sequential
    models.  First, let's say that you have a Sequential model, and you want to
    freeze all layers except the last one. In this case, you would simply
    iterate over model.layers and set layer.trainable = False on each layer,
    except the last one.

    Another common blueprint is to use a Sequential model to stack a
    pre-trained model and some freshly initialized classification layers.
    '''
    print(__doc__)

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
