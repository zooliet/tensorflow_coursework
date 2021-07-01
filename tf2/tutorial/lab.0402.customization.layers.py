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


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - Layers: common sets of useful operations
if args.step == 1:
    print("\n### Step #1 - Layers: common sets of useful operations")

    # In the tf.keras.layers package, layers are objects. To construct a layer,
    # simply construct the object. Most layers take as a first argument the number
    # of output dimensions / channels.
    layer = tf.keras.layers.Dense(100)

    # The number of input dimensions is often unnecessary, as it can be inferred
    # the first time the layer is used, but it can be provided if you want to
    # specify it manually, which is useful in some complex models.
    layer = tf.keras.layers.Dense(10, input_shape=(None, 5))

    logger.info('To use a layer, simply call it')
    print(layer(tf.zeros([10, 5])), '\n')

    # Layers have many useful methods. For example, you can inspect all variables
    # in a layer using `layer.variables` and trainable variables using
    # `layer.trainable_variables`. In this case a fully-connected layer
    # will have variables for weights and biases.
    logger.info('layer.variables:')
    print(layer.variables, '\n')

    logger.info('The variables are also accessible through nice accessors:')
    print(layer.kernel) 
    print(layer.bias)


args.step = auto_increment(args.step, args.all)
### Step #2 - Implementing custom layers
if args.step == 2:
    print("\n### Step #2 - Implementing custom layers")

    class MyDenseLayer(tf.keras.layers.Layer):
        def __init__(self, num_outputs):
            super(MyDenseLayer, self).__init__()
            self.num_outputs = num_outputs

        def build(self, input_shape):
            self.kernel = self.add_weight(
                "kernel",
                shape=[int(input_shape[-1]), self.num_outputs]
            )

        def call(self, inputs):
            return tf.matmul(inputs, self.kernel)

    layer = MyDenseLayer(10)

    _ = layer(tf.zeros([10, 5])) # Calling the layer `.builds` it.
    logger.info('layer.trainable_variables:')
    print([var.name for var in layer.trainable_variables])


args.step = auto_increment(args.step, args.all)
### Step #3 - Models: Composing layers
if args.step == 3:
    print("\n### Step #3 - Models: Composing layers")

    class ResnetIdentityBlock(tf.keras.Model):
        def __init__(self, kernel_size, filters):
            super(ResnetIdentityBlock, self).__init__(name='')
            filters1, filters2, filters3 = filters

            self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
            self.bn2a = tf.keras.layers.BatchNormalization()

            self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
            self.bn2b = tf.keras.layers.BatchNormalization()

            self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
            self.bn2c = tf.keras.layers.BatchNormalization()

        def call(self, input_tensor, training=False):
            x = self.conv2a(input_tensor)
            x = self.bn2a(x, training=training)
            x = tf.nn.relu(x)

            x = self.conv2b(x)
            x = self.bn2b(x, training=training)
            x = tf.nn.relu(x)

            x = self.conv2c(x)
            x = self.bn2c(x, training=training)

            x += input_tensor
            return tf.nn.relu(x)

    block = ResnetIdentityBlock(1, [1, 2, 3])

    _ = block(tf.zeros([1, 2, 3, 3]))
    logger.info('block.layers:')
    print(*[layer for layer in block.layers], sep='\n')
    
    logger.info(f'len(block.variables): {len(block.variables)}')
    block.summary()

    my_seq = tf.keras.Sequential([
        tf.keras.layers.Conv2D(1, (1, 1), input_shape=(None, None, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(2, 1, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(3, (1, 1)),
        tf.keras.layers.BatchNormalization()
    ])

    logger.info('my_seq(tf.zeros([1, 2, 3, 3])):')
    print(my_seq(tf.zeros([1, 2, 3, 3])), '\n')
    my_seq.summary()


### End of File
print()
if args.plot:
    plt.show()
debug()


