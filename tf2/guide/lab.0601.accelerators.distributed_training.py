#!/usr/bin/env python

import sys
sys.path.append('./')
sys.path.append('../../')

from lab_utils import (
    tf, os, np, plt, logger, ap, BooleanAction,
    debug, toc, auto_increment
)

# ap.add_argument('--epochs', type=int, default=2, help='number of epochs: 2*')
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


args.step = auto_increment(args.step, args.all)
### Step #1 - Overview
if args.step == 1:
    print("\n### Step #1 - Overview")
    
    __doc__= '''
    tf.distribute.Strategy is a TensorFlow API to distribute training across
    multiple GPUs, multiple machines or TPUs. Using this API, you can
    distribute your existing models and training code with minimal code
    changes.

    tf.distribute.Strategy can be used with a high-level API like Keras, and
    can also be used to distribute custom training loops (and, in general, any
    computation using TensorFlow).

    In TensorFlow 2.x, you can execute your programs eagerly, or in a graph
    using tf.function. tf.distribute.Strategy intends to support both these
    modes of execution, but works best with tf.function. Eager mode is only
    recommended for debugging purpose and not supported for TPUStrategy.

    You can use tf.distribute.Strategy with very few changes to your code,
    because the underlying components of TensorFlow have been changed to become
    strategy-aware. This includes variables, layers, models, optimizers,
    metrics, summaries, and checkpoints.
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #2 - Types of strategies
if args.step == 2:
    print("\n### Step #2 - Types of strategies")

    __doc__='''
    - Synchronous vs asynchronous training: These are two common ways of
      distributing training with data parallelism. In sync training, all
      workers train over different slices of input data in sync, and
      aggregating gradients at each step. In async training, all workers are
      independently training over the input data and updating variables
      asynchronously. Typically sync training is supported via all-reduce and
      async through parameter server architecture.
    - Hardware platform: You may want to scale your training onto multiple GPUs
      on one machine, or multiple machines in a network (with 0 or more GPUs
      each), or on Cloud TPUs.
    '''
    print(__doc__)

    if args.plot:
        plt.figure()
        img = tf.io.read_file('supplement/tf2_g0601_01.png')
        img = tf.image.decode_png(img)
        plt.imshow(img)
        plt.show(block=False)


### End of File
print()
if args.plot:
    plt.show()
debug()

