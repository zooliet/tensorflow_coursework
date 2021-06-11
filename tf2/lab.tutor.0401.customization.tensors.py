#!/usr/bin/env python

import sys
sys.path.append('./')
sys.path.append('../')

from lab_utils import (
    os, np, plt, logger, ap, BooleanAction,
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

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("\n#################################################")
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - Tensors
if args.step == 1:
    print("\n### Step #1 - Tensors")

    logger.info('tf.add(1,2)')
    print(tf.add(1, 2), '\n')

    logger.info('tf.add([1, 2], [3, 4])')
    print(tf.add([1, 2], [3, 4]), '\n')

    logger.info('tf.square(5)')
    print(tf.square(5), '\n')
    
    logger.info('tf.reduce_sum([1, 2, 3])')
    print(tf.reduce_sum([1, 2, 3]), '\n')

    # Operator overloading is also supported
    logger.info('tf.square(2) + tf.square(3)')
    print(tf.square(2) + tf.square(3), '\n')

    logger.info('tf.matmul([[1]], [[2, 3]])')
    x = tf.matmul([[1]], [[2, 3]])
    print(x, '\n')


args.step = auto_increment(args.step, args.all)
### Step #2 - Tensors: NumPy Compatibility
if args.step == 2:
    print("\n### Step #2 - Tensors: NumPy Compatibility")

    ndarray = np.ones([3, 3])

    logger.info("TensorFlow operations convert numpy arrays to Tensors")
    tensor = tf.multiply(ndarray, 42)
    print(tensor, '\n')

    logger.info("And NumPy operations convert Tensors to numpy arrays")
    print(np.add(tensor, 1), '\n')

    logger.info("The .numpy() explicitly converts a Tensor to a numpy array")
    print(tensor.numpy(), '\n')


args.step = auto_increment(args.step, args.all)
### Step #3 - GPU acceleration
if args.step == 3:
    print("\n### Step #3 - GPU acceleration")

    x = tf.random.uniform([3, 3])

    logger.info("Is there a GPU available: "),
    print(tf.config.list_physical_devices("GPU"), '\n')

    logger.info("Is the Tensor on GPU #0:  "),
    print(x.device.endswith('GPU:0'), '\n')


args.step = auto_increment(args.step, args.all)
### Step #4 - GPU acceleration: Device Names
if args.step == 4:
    print("\n### Step #4 - GPU acceleration: Device Names")


args.step = auto_increment(args.step, args.all)
### Step #5 - GPU acceleration: Explicit Device Placement
if args.step == 5:
    print("\n### Step #5 - GPU acceleration: Explicit Device Placement")

    def time_matmul(x, n=10):
        start = time.time()
        for loop in range(n):
            tf.matmul(x, x)

        result = time.time()-start
        print("{} loops: {:0.2f}ms".format(n, 1000*result))

    N = 1000 # 100, 10(작은 반복에서는 CPU가 더 빠름)

    # Force execution on CPU
    logger.info("On CPU:")
    with tf.device("CPU:0"):
        x = tf.random.uniform([1000, 1000])
        assert x.device.endswith("CPU:0")
        time_matmul(x, n=N)

    # Force execution on GPU #0 if available
    if tf.config.list_physical_devices("GPU"):
        logger.info("On GPU:")
        with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
            x = tf.random.uniform([1000, 1000])
            assert x.device.endswith("GPU:0")
            time_matmul(x, n=N)
    

### End of File
if args.plot:
    plt.show()
debug()


