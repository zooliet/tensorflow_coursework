#!/usr/bin/env python

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


### TOC
if args.step == 0:
    toc(__file__)


if args.step or args.all:
    class Net(tf.keras.Model):
        """A simple linear model."""

        def __init__(self):
            super(Net, self).__init__()
            self.l1 = tf.keras.layers.Dense(5)

        def call(self, x):
            return self.l1(x)


args.step = auto_increment(args.step, args.all)
### Step #1 - Training checkpoints
if args.step == 1:
    print("\n### Step #1 - Training checkpoints")

    __doc__ = '''
    The phrase "Saving a TensorFlow model" typically means one of two things:
    1. Checkpoints, OR 2. SavedModel.

    Checkpoints capture the exact value of all parameters (tf.Variable objects)
    used by a model. Checkpoints do not contain any description of the
    computation defined by the model and thus are typically only useful when
    source code is available.

    The SavedModel format on the other hand includes a serialized description
    of the computation defined by the model in addition to the parameter values
    (checkpoint). Models in this format are independent of the source code.
    They are thus suitable for deployment via TensorFlow Serving, TensorFlow
    Lite, TensorFlow.js, or programs in other programming languages.
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #2 - Saving from tf.keras training APIs
if args.step == 2:
    print("\n### Step #2 - Saving from tf.keras training APIs")

    net = Net()
    net.save_weights('tmp/tf2_g0501/easy_checkpoint')

args.step = auto_increment(args.step, args.all)
### Step #3 - Writing checkpoints: Manual checkpointing
if args.step == 3:
    print("\n### Step #3 - Writing checkpoints: Manual checkpointing")

    __doc__='''
    The persistent state of a TensorFlow model is stored in tf.Variable
    objects. These can be constructed directly, but are often created through
    high-level APIs like tf.keras.layers or tf.keras.Model.

    The easiest way to manage variables is by attaching them to Python objects,
    then referencing those objects.

    Subclasses of tf.train.Checkpoint, tf.keras.layers.Layer, and
    tf.keras.Model automatically track variables assigned to their attributes.
    The following example constructs a simple linear model, then writes
    checkpoints which contain values for all of the model's variables.

    You can easily save a model-checkpoint with Model.save_weights.
    '''
    print(__doc__)

    net = Net()

    def toy_dataset():
        inputs = tf.range(10.)[:, None]
        labels = inputs * 5. + tf.range(5.)[None, :]

        return tf.data.Dataset.from_tensor_slices(
            dict(x=inputs, y=labels)
        ).repeat().batch(2)

    def train_step(net, example, optimizer):
        """Trains `net` on `example` using `optimizer`."""
        with tf.GradientTape() as tape:
            output = net(example['x'])
            loss = tf.reduce_mean(tf.abs(output - example['y']))
        variables = net.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss    

    # Create the checkpoint objects
    opt = tf.keras.optimizers.Adam(0.1)
    dataset = toy_dataset()
    iterator = iter(dataset)

    ckpt = tf.train.Checkpoint(
        step=tf.Variable(1), 
        optimizer=opt, 
        net=net, 
        iterator=iterator
    )
    manager = tf.train.CheckpointManager(ckpt, 'tmp/tf2_g0501/tf_ckpts', max_to_keep=3)

    # Train and checkpoint the model
    def train_and_checkpoint(net, manager):
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            logger.info("Restored from {}\n".format(manager.latest_checkpoint))
        else:
            logger.info("Initializing from scratch.\n")

        for _ in range(50):
            example = next(iterator)
            loss = train_step(net, example, opt)
            ckpt.step.assign_add(1)
            if int(ckpt.step) % 10 == 0:
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                print("loss {:1.2f}".format(loss.numpy()))
        print()

    train_and_checkpoint(net, manager)

    # Restore and continue training
    opt = tf.keras.optimizers.Adam(0.1)
    net = Net()
    dataset = toy_dataset()
    iterator = iter(dataset)
    ckpt = tf.train.Checkpoint(
        step=tf.Variable(1), 
        optimizer=opt, 
        net=net, 
        iterator=iterator
    )
    manager = tf.train.CheckpointManager(ckpt, 'tmp/tf2_g0501/tf_ckpts', max_to_keep=3)
    train_and_checkpoint(net, manager)

    logger.info('List the three remaining checkpoints:') 
    print(*manager.checkpoints, sep='\n') 


args.step = auto_increment(args.step, args.all)
### Step #4 - Loading mechanics
if args.step == 4:
    print("\n### Step #4 - Loading mechanics")
    pass


### End of File
print()
if args.plot:
    plt.show()
debug()

