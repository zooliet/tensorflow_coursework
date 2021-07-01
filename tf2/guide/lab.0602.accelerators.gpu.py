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
    TensorFlow supports running computations on a variety of types of devices,
    including CPU and GPU. They are represented with string identifiers for
    example:

    - "/device:CPU:0": The CPU of your machine.
    - "/GPU:0": Short-hand notation for the first GPU of your machine that is
      visible to TensorFlow.
    - "/job:localhost/replica:0/task:0/device:GPU:1": Fully qualified name of
      the second GPU of your machine that is visible to TensorFlow.

    If a TensorFlow operation has both CPU and GPU implementations, by default
    the GPU devices will be given priority when the operation is assigned to a
    device. For example, tf.matmul has both CPU and GPU kernels. On a system
    with devices CPU:0 and GPU:0, the GPU:0 device will be selected to run
    tf.matmul unless you explicitly request running it on another device.
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #2 - Logging device placement
if args.step == 2:
    print("\n### Step #2 - Logging device placement")

    __doc__='''
    To find out which devices your operations and tensors are assigned to, put
    tf.debugging.set_log_device_placement(True) as the first statement of your
    program. Enabling device placement logging causes any Tensor allocations or
    operations to be printed.
    '''
    print(__doc__)

    tf.debugging.set_log_device_placement(True)


args.step = auto_increment(args.step, args.all)
### Step #3 - Manual device placement
if args.step == 3:
    print("\n### Step #3 - Manual device placement")

    # Place tensors on the CPU
    with tf.device('/CPU:0'):
      a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # Run on the GPU
    c = tf.matmul(a, b)


args.step = auto_increment(args.step, args.all)
### Step #4 - Limiting GPU memory growth
if args.step == 4:
    print("\n### Step #4 - Limiting GPU memory growth")

    __doc__='''
    By default, TensorFlow maps nearly all of the GPU memory of all GPUs
    (subject to CUDA_VISIBLE_DEVICES) visible to the process. This is done to
    more efficiently use the relatively precious GPU memory resources on the
    devices by reducing memory fragmentation. To limit TensorFlow to a specific
    set of GPUs we use the tf.config.experimental.set_visible_devices method.
    '''
    print(__doc__)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            logger.info(f'{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPU')
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    __doc__='''
    In some cases it is desirable for the process to only allocate a subset of
    the available memory, or to only grow the memory usage as is needed by the
    process. TensorFlow provides two methods to control this.

    The first option is to turn on memory growth by calling
    tf.config.experimental.set_memory_growth, which attempts to allocate only
    as much GPU memory as needed for the runtime allocations: it starts out
    allocating very little memory, and as the program gets run and more GPU
    memory is needed, we extend the GPU memory region allocated to the
    TensorFlow process. Note we do not release memory, since it can lead to
    memory fragmentation. To turn on memory growth for a specific GPU, use the
    following code prior to allocating any tensors or executing any ops.

    Another way to enable this option is to set the environmental variable
    TF_FORCE_GPU_ALLOW_GROWTH to true. This configuration is platform specific.
    '''
    print(__doc__)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            logger.info(f'{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPU')
        except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
            print(e)


args.step = auto_increment(args.step, args.all)
### Step #5 - Using a single GPU on a multi-GPU system
if args.step == 5:
    print("\n### Step #5 - Using a single GPU on a multi-GPU system")

    __doc__='''
    If you have more than one GPU in your system, the GPU with the lowest ID
    will be selected by default. If you would like to run on a different GPU,
    you will need to specify the preference explicitly.
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #6 - Using multiple GPUs
if args.step == 6:
    print("\n### Step #6 - Using multiple GPUs")

    __doc__='''
    Developing for multiple GPUs will allow a model to scale with the
    additional resources. If developing on a system with a single GPU, we can
    simulate multiple GPUs with virtual devices. This enables easy testing of
    multi-GPU setups without requiring additional resources.
    '''
    print(__doc__)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Create 2 virtual GPUs with 1GB memory each
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
                 tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            logger.info(f'{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPU\n')
        except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
            print(e)

    __doc__='''
    Once we have multiple logical GPUs available to the runtime, we can utilize
    the multiple GPUs with tf.distribute.Strategy or with manual placement. The
    best practice for using multiple GPUs is to use tf.distribute.Strategy.

    This program will run a copy of your model on each GPU, splitting the input
    data between them, also known as "data parallelism".
    tf.debugging.set_log_device_placement(True)
    '''
    print(__doc__)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        inputs = tf.keras.layers.Input(shape=(1,))
        predictions = tf.keras.layers.Dense(1)(inputs)
        model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
        model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.2)
        )    

    # tf.distribute.Strategy works under the hood by replicating computation
    # across devices. You can manually implement replication by constructing
    # your model on each GPU. For example:
    tf.debugging.set_log_device_placement(True)

    gpus = tf.config.experimental.list_logical_devices('GPU')
    if gpus:
        # Replicate your computation on multiple GPUs
        c = []
        for gpu in gpus:
            with tf.device(gpu.name):
                a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                c.append(tf.matmul(a, b))

        with tf.device('/CPU:0'):
            matmul_sum = tf.add_n(c)

    print(matmul_sum)


### End of File
print()
if args.plot:
    plt.show()
debug()

