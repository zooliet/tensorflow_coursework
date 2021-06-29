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
import timeit
import datetime


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - Extract tensor slices
if args.step == 1:
    print("\n### Step #1 - Extract tensor slices")

    t1 = tf.constant([0, 1, 2, 3, 4, 5, 6, 7])
    logger.info('using tf.slice():')  
    print(tf.slice(t1, begin=[1], size=[3]), '\n')
    
    logger.info('using a more Pythonic syntax:')  
    print(t1[1:4], '\n')

    if args.plot:
        plt.figure()
        img = tf.io.read_file('supplement/tf2_g0112_01.png')
        img = tf.image.decode_png(img)
        plt.imshow(img)
        plt.show(block=False)
        
    print(t1[-3:], '\n')
    if args.plot:
        plt.figure()
        img = tf.io.read_file('supplement/tf2_g0112_02.png')
        img = tf.image.decode_png(img)
        plt.imshow(img)
        plt.show(block=False)

    t2 = tf.constant([[0, 1, 2, 3, 4],
                      [5, 6, 7, 8, 9],
                      [10, 11, 12, 13, 14],
                      [15, 16, 17, 18, 19]])
    
    logger.info('For 2-dimensional tensors,you can use something like:')
    print(t2[:-1, 1:3], '\n')

    if args.plot:
        plt.figure()
        img = tf.io.read_file('supplement/tf2_g0112_03.png')
        img = tf.image.decode_png(img)
        plt.imshow(img)
        plt.show(block=False)

    t3 = tf.constant([
        [
            [1, 3, 5, 7], [9, 11, 13, 15]
        ],
        [
            [17, 19, 21, 23], [25, 27, 29, 31]
        ]
    ])

    logger.info('use tf.slice on higher dimensional tensors as well:')
    print(tf.slice(t3, begin=[1, 1, 0], size=[1, 1, 2]), '\n')
    
    logger.info('Use tf.gather to extract specific indices from a single axis of a tensor:')
    print(tf.gather(t1, indices=[0, 3, 6]), '\n')
    # This is similar to doing
    print(t1[::3], '\n')

    if args.plot:
        plt.figure()
        img = tf.io.read_file('supplement/tf2_g0112_04.png')
        img = tf.image.decode_png(img)
        plt.imshow(img)
        plt.show(block=False)

    logger.info('tf.gather does not require indices to be evenly spaced:')
    alphabet = tf.constant(list('abcdefghijklmnopqrstuvwxyz'))
    print(tf.gather(alphabet, indices=[2, 0, 19, 18]), '\n')

    if args.plot:
        plt.figure()
        img = tf.io.read_file('supplement/tf2_g0112_05.png')
        img = tf.image.decode_png(img)
        plt.imshow(img)
        plt.show(block=False)

    logger.info('To extract slices from multiple axes of a tensor, use tf.gather_nd:')
    t4 = tf.constant([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])
    print(tf.gather_nd(t4, indices=[[2], [3], [0]]), '\n')

    if args.plot:
        plt.figure()
        img = tf.io.read_file('supplement/tf2_g0112_06.png')
        img = tf.image.decode_png(img)
        plt.imshow(img)
        plt.show(block=False)

    t5 = np.reshape(np.arange(18), [2, 3, 3])
    print(tf.gather_nd(t5, indices=[[0, 0, 0], [1, 2, 1]]), '\n')

    logger.info('Return a list of two matrices:')
    print(tf.gather_nd(t5, indices=[[[0, 0], [0, 2]], [[1, 0], [1, 2]]]), '\n')

    logger.info('Return one matrix:')
    print(tf.gather_nd(t5, indices=[[0, 0], [0, 2], [1, 0], [1, 2]]), '\n')


args.step = auto_increment(args.step, args.all)
### Step #2 - Insert data into tensors
if args.step == 2:
    print("\n### Step #2 - Insert data into tensors")

    logger.info('Use tf.scatter_nd to insert data at specific slices/indices of a tensor:')
    t6 = tf.constant([10])
    indices = tf.constant([[1], [3], [5], [7], [9]])
    data = tf.constant([2, 4, 6, 8, 10])

    print(tf.scatter_nd(indices=indices, updates=data, shape=t6), '\n')

    logger.info('Gather values from one tensor by specifying indices:')
    new_indices = tf.constant([[0, 2], [2, 1], [3, 3]])
    t2 = tf.constant([
        [0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]
    ])
    t7 = tf.gather_nd(t2, indices=new_indices)
    print(t7, '\n')
    if args.plot:
        plt.figure()
        img = tf.io.read_file('supplement/tf2_g0112_07.png')
        img = tf.image.decode_png(img)
        plt.imshow(img)
        plt.show(block=False)

    logger.info('Add these values into a new tensor:')
    t8 = tf.scatter_nd(indices=new_indices, updates=t7, shape=tf.constant([4, 5]))
    print(t8, '\n')

    logger.info('This is similar to:')
    t9 = tf.SparseTensor(
        indices=[[0, 2], [2, 1], [3, 3]], values=[2, 11, 18], dense_shape=[4, 5]
    )
    print(t9, '\n')

    logger.info('Convert the sparse tensor into a dense tensor:')
    t10 = tf.sparse.to_dense(t9)
    print(t10, '\n')

    str='''
    To insert data into a tensor with pre-existing values, 
    use tf.tensor_scatter_nd_add:
    '''
    logger.info(str)

    t11 = tf.constant([
        [2, 7, 0], [9, 0, 1], [0, 3, 8]
    ])

    # Convert the tensor into a magic square by inserting numbers at appropriate indices
    t12 = tf.tensor_scatter_nd_add(
        t11,
        indices=[[0, 2], [1, 1], [2, 0]],
        updates=[6, 5, 4]
    )
    print(t12, '\n')

    logger.info('use tf.tensor_scatter_nd_sub to subtract values from a tensor with pre-existing values:')
    # Convert the tensor into an identity matrix
    t13 = tf.tensor_scatter_nd_sub(
        t11,
        indices=[[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 1], [2, 2]],
        updates=[1, 7, 9, -1, 1, 3, 7]
    )
    print(t13, '\n')

    logger.info('Use tf.tensor_scatter_nd_min to copy element-wise minimum values from one tensor to another:')
    t14 = tf.constant([
        [-2, -7, 0], [-9, 0, 1], [0, -3, -8]
    ])
    t15 = tf.tensor_scatter_nd_min(
        t14,
        indices=[[0, 2], [1, 1], [2, 0]],
        updates=[-6, -5, -4]
    )
    print(t15, '\n')

    logger.info('Use tf.tensor_scatter_nd_max to copy element-wise maximum values from one tensor to another:')
    t16 = tf.tensor_scatter_nd_max(
        t14,
        indices=[[0, 2], [1, 1], [2, 0]],
        updates=[6, 5, 4]
    )
    print(t16, '\n')


### End of File
if args.plot:
    plt.show()
debug()
