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


if args.step or args.all:
    # artificial dataset
    class ArtificialDataset(tf.data.Dataset):
        def _generator(num_samples):
            # Opening the file
            time.sleep(0.03)

            for sample_idx in range(num_samples):
                # Reading data (line, record) from the file
                time.sleep(0.015)

                yield (sample_idx,)

        def __new__(cls, num_samples=3):
            return tf.data.Dataset.from_generator(
                cls._generator,
                output_signature = tf.TensorSpec(shape = (1,), dtype = tf.int64),
                args=(num_samples,)
            )

    # dummy training loop that measures how long it takes to iterate over a
    # dataset
    def benchmark(dataset, num_epochs=2):
        start_time = time.perf_counter()
        for epoch_num in range(num_epochs):
            for sample in dataset:
                # Performing a training step
                time.sleep(0.01)
        print(f"Execution time: {time.perf_counter() - start_time:.4f} sec")


args.step = auto_increment(args.step, args.all)
### Step #1 - Overview
if args.step == 1:
    print("\n### Step #1 - Overview")
    
    __doc__= '''
    GPUs and TPUs can radically reduce the time required to execute a single
    training step. Achieving peak performance requires an efficient input
    pipeline that delivers data for the next step before the current step has
    finished. The tf.data API helps to build flexible and efficient input
    pipelines. This document demonstrates how to use the tf.data API to build
    highly performant TensorFlow input pipelines.
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #2 - Optimize performance: The naive approach 
if args.step == 2:
    print("\n### Step #2 - Optimize performance: The naive approach")

    logger.info('ArtificationDataset():')
    benchmark(ArtificialDataset())

    if args.plot:
        plt.figure()
        img = tf.io.read_file('supplement/tf2_g0402_01.png')
        img = tf.image.decode_png(img)
        plt.imshow(img)
        plt.show(block=False)
        

args.step = auto_increment(args.step, args.all)
### Step #3 - Optimize performance: Prefetching
if args.step == 3:
    print("\n### Step #3 - Optimize performance: Prefetching")

    __doc__='''
    Prefetching overlaps the preprocessing and model execution of a training
    step. While the model is executing training step s, the input pipeline is
    reading the data for step s+1. Doing so reduces the step time to the
    maximum (as opposed to the sum) of the training and the time it takes to
    extract the data.

    The tf.data API provides the tf.data.Dataset.prefetch transformation. It
    can be used to decouple the time when data is produced from the time when
    data is consumed. In particular, the transformation uses a background
    thread and an internal buffer to prefetch elements from the input dataset
    ahead of the time they are requested. The number of elements to prefetch
    should be equal to (or possibly greater than) the number of batches
    consumed by a single training step. You could either manually tune this
    value, or set it to tf.data.AUTOTUNE, which will prompt the tf.data runtime
    to tune the value dynamically at runtime.
    '''
    print(__doc__)

    logger.info('ArtificialDataset().prefetch(tf.data.AUTOTUNE):') 
    benchmark(ArtificialDataset().prefetch(tf.data.AUTOTUNE))

    if args.plot:
        plt.figure()
        img = tf.io.read_file('supplement/tf2_g0402_02.png')
        img = tf.image.decode_png(img)
        plt.imshow(img)
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #4 - Optimize performance: Parallelizing data extraction
if args.step == 4:
    print("\n### Step #4 - Reading input data: Parallelizing data extraction")

    # Sequential interleave
    # The default arguments of the tf.data.Dataset.interleave transformation
    # make it interleave single samples from two datasets sequentially.
    logger.info('tf.data.Dataset.range(2).interleave():')
    benchmark(
        tf.data.Dataset.range(2).interleave(lambda _: ArtificialDataset())
    )

    # Parallel interleave
    # Now, use the num_parallel_calls argument of the interleave
    # transformation. This loads multiple datasets in parallel, reducing the
    # time waiting for the files to be opened.
    logger.info('tf.data.Dataset.range(2).interleave(num_parallel_calls):')
    benchmark(
        tf.data.Dataset.range(2)
        .interleave(
            lambda _: ArtificialDataset(),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    )

    if args.plot:
        plt.figure()
        img = tf.io.read_file('supplement/tf2_g0402_03.png')
        img = tf.image.decode_png(img)
        plt.imshow(img)
        plt.show(block=False)

        plt.figure()
        img = tf.io.read_file('supplement/tf2_g0402_04.png')
        img = tf.image.decode_png(img)
        plt.imshow(img)
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #5 - Optimize performance: Parallelizing data transformation
if args.step in [5, 6]:
    print("\n### Step #5 - Reading input data: Parallelizing data transformation")

    __doc__='''
    When preparing data, input elements may need to be pre-processed. To this
    end, the tf.data API offers the tf.data.Dataset.map transformation, which
    applies a user-defined function to each element of the input dataset.
    Because input elements are independent of one another, the pre-processing
    can be parallelized across multiple CPU cores. To make this possible,
    similarly to the prefetch and interleave transformations, the map
    transformation provides the num_parallel_calls argument to specify the
    level of parallelism.

    Choosing the best value for the num_parallel_calls argument depends on your
    hardware, characteristics of your training data (such as its size and
    shape), the cost of your map function, and what other processing is
    happening on the CPU at the same time. A simple heuristic is to use the
    number of available CPU cores. However, as for the prefetch and interleave
    transformation, the map transformation supports tf.data.AUTOTUNE which will
    delegate the decision about what level of parallelism to use to the tf.data
    runtime.
    '''
    if args.step == 5: print(__doc__)

    def mapped_function(s):
        # Do some hard pre-processing
        tf.py_function(lambda: time.sleep(0.03), [], ())
        return s

    # Sequential mapping
    # Start by using the map transformation without parallelism as a baseline example.
    if args.step == 5:
        logger.info('ArtificialDataset().map(mapped_function):')
        benchmark(ArtificialDataset().map(mapped_function))

        # Parallel mapping
        # Now, use the same pre-processing function but apply it in parallel on
        # multiple samples.

        logger.info('ArtificialDataset().map(num_parallel_calls):')
        benchmark(
            ArtificialDataset().map(
                mapped_function,
                num_parallel_calls=tf.data.AUTOTUNE
            )
        )    
        if args.plot:
            plt.figure()
            img = tf.io.read_file('supplement/tf2_g0402_05.png')
            img = tf.image.decode_png(img)
            plt.imshow(img)
            plt.show(block=False)

            plt.figure()
            img = tf.io.read_file('supplement/tf2_g0402_06.png')
            img = tf.image.decode_png(img)
            plt.imshow(img)
            plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #6 - Caching
if args.step == 6:
    print("\n### Step #6 - Caching")

    __doc__='''
    The tf.data.Dataset.cache transformation can cache a dataset, either in
    memory or on local storage. This will save some operations (like file
    opening and data reading) from being executed during each epoch.

    If the user-defined function passed into the map transformation is
    expensive, apply the cache transformation after the map transformation as
    long as the resulting dataset can still fit into memory or local storage.
    If the user-defined function increases the space required to store the
    dataset beyond the cache capacity, either apply it after the cache
    transformation or consider pre-processing your data before your training
    job to reduce resource usage.
    '''
    print(__doc__)

    logger.info('ArtificialDataset().map().cache():')
    benchmark(
        ArtificialDataset().map(  
            # Apply time consuming operations before cache
            mapped_function
        ).cache(), 5
    )

    if args.plot:
        plt.figure()
        img = tf.io.read_file('supplement/tf2_g0402_07.png')
        img = tf.image.decode_png(img)
        plt.imshow(img)
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #7 - Vectorizing mapping
if args.step == 7:
    print("\n### Step #7 - Vectorizing mapping")

    __doc__='''
    Invoking a user-defined function passed into the map transformation has
    overhead related to scheduling and executing the user-defined function.
    Vectorize the user-defined function (that is, have it operate over a batch
    of inputs at once) and apply the batch transformation before the map
    transformation.

    To illustrate this good practice, your artificial dataset is not suitable.
    The scheduling delay is around 10 microseconds (10e-6 seconds), far less
    than the tens of milliseconds used in the ArtificialDataset, and thus its
    impact is hard to see.
    '''
    print(__doc__)

    fast_dataset = tf.data.Dataset.range(10000)

    def fast_benchmark(dataset, num_epochs=2):
        start_time = time.perf_counter()
        for _ in tf.data.Dataset.range(num_epochs):
            for _ in dataset:
                pass
        tf.print(f"Execution time: {time.perf_counter() - start_time:.4f} sec")

    def increment(x):
        return x+1

    # Scalar mapping
    logger.info('fast_dataset.map().batch():')
    fast_benchmark(
        fast_dataset.map(increment) # Apply function one item at a time
        .batch(256) # Batch
    )

    # Vectorized mapping
    logger.info('fast_dataset.batch().map():')
    fast_benchmark(
        fast_dataset.batch(256)
        # Apply function on a batch of items
        # The tf.Tensor.__add__ method already handle batches
        .map(increment)
    )

    if args.plot:
        plt.figure()
        img = tf.io.read_file('supplement/tf2_g0402_08.png')
        img = tf.image.decode_png(img)
        plt.imshow(img)
        plt.show(block=False)

        plt.figure()
        img = tf.io.read_file('supplement/tf2_g0402_09.png')
        img = tf.image.decode_png(img)
        plt.imshow(img)
        plt.show(block=False)


### Step #8 - Reducing memory footprint()
if args.step == 8:
    print("\n### Step #8 - Reducing memory footprint")

    __doc__='''
    A number of transformations, including interleave, prefetch, and shuffle,
    maintain an internal buffer of elements. If the user-defined function
    passed into the map transformation changes the size of the elements, then
    the ordering of the map transformation and the transformations that buffer
    elements affects the memory usage. In general, choose the order that
    results in lower memory footprint, unless different ordering is desirable
    for performance.

    It is recommended to cache the dataset after the map transformation except
    if this transformation makes the data too big to fit in memory. A trade-off
    can be achieved if your mapped function can be split in two parts: a time
    consuming one and a memory consuming part. In this case, you can chain your
    transformations like below:

    dataset.map(time_consuming_mapping).cache().map(memory_consuming_mapping)
    '''
    print(__doc__)


### Step #9 - Best practice summary
if args.step == 9:
    print("\n### Step #9 - Best practice summary")

    __doc__='''
    - Use the prefetch transformation to overlap the work of a producer and
      consumer
    - Parallelize the data reading transformation using the interleave
      transformation
    - Parallelize the map transformation by setting the num_parallel_calls
      argument
    - Use the cache transformation to cache data in memory during the first
      epoch
    - Vectorize user-defined functions passed in to the map transformation
    - Reduce memory usage when applying the interleave, prefetch, and shuffle
      transformations
    '''
    print(__doc__)

    
### End of File
print()
if args.plot:
    plt.show()
debug()
