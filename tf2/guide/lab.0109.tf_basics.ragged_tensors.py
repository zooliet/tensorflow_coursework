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
### Step #1 - Overview
if args.step == 1:
    print("\n### Step #1 - Overview")

    __doc__='''
    Your data comes in many shapes; your tensors should too. Ragged tensors are
    the TensorFlow equivalent of nested variable-length lists. They make it
    easy to store and process data with non-uniform shapes, including:
    - Variable-length features, such as the set of actors in a movie.
    - Batches of variable-length sequential inputs, such as sentences or video
      clips.
    - Hierarchical inputs, such as text documents that are subdivided into
      sections, paragraphs, sentences, and words.
    - Individual fields in structured inputs, such as protocol buffers.
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #2 - Overview: What you can do with a ragged tensor
if args.step == 2:
    print("\n### Step #2 - Overview: What you can do with a ragged tensor")

    digits = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
    words = tf.ragged.constant([["So", "long"], ["thanks", "for", "all", "the", "fish"]])

    logger.info('tf.add(digits, 3):')
    print(tf.add(digits, 3), '\n')

    logger.info('tf.reduce_mean(digits, axis=1):')
    print(tf.reduce_mean(digits, axis=1), '\n')

    logger.info('tf.concat([digits, [[5, 3]]], axis=0):')
    print(tf.concat([digits, [[5, 3]]], axis=0), '\n')

    logger.info('tf.tile(digits, [1, 2]):')
    print(tf.tile(digits, [1, 2]), '\n')

    logger.info('tf.strings.substr(words, 0, 2):')
    print(tf.strings.substr(words, 0, 2), '\n')

    logger.info('tf.map_fn(tf.math.square, digits):')
    print(tf.map_fn(tf.math.square, digits), '\n')

    # As with normal tensors, you can use Python-style indexing to access specific slices 
    # of a ragged tensor.
    logger.info('digits[0]:') # First row
    print(digits[0], '\n')  
    
    logger.info('digits[:, :2]:')  # First two values in each row.
    print(digits[:, :2], '\n')  

    logger.info('digits[:, -2:]:')  # Last two values in each row.
    print(digits[:, -2:], '\n')  

    # And just like normal tensors, you can use Python arithmetic and comparison operators 
    # to perform elementwise operations. 
    logger.info('digits + 3:')
    print(digits + 3, '\n')

    logger.info('digits + tf.ragged.constant([[1, 2, 3, 4], [], [5, 6, 7], [8], []]):')
    print(digits + tf.ragged.constant([[1, 2, 3, 4], [], [5, 6, 7], [8], []]), '\n')

    # If you need to perform an elementwise transformation to the values of a RaggedTensor, 
    # you can use tf.ragged.map_flat_values, which takes a function plus one or more 
    # arguments, and applies the function to transform the RaggedTensor's values.
    
    logger.info('use tf.ragged.map_flat_values(fn, RaggedTensor):')
    times_two_plus_one = lambda x: x * 2 + 1
    print(tf.ragged.map_flat_values(times_two_plus_one, digits), '\n')
    # print(tf.map_fn(times_two_plus_one, digits), '\n')

    logger.info('Ragged tensors can be converted to nested Python lists and NumPy arrays:')
    print(digits.to_list(), '\n')
    print(digits.numpy())


args.step = auto_increment(args.step, args.all)
### Step #3 - Overview: Constructing a ragged tensor
if args.step == 3:
    print("\n### Step #3 - Overview: Constructing a ragged tensor")

    sentences = tf.ragged.constant([
        ["Let's", "build", "some", "ragged", "tensors", "!"],
        ["We", "can", "use", "tf.ragged.constant", "."]
    ])
    logger.info('using tf.ragged.constant():')
    print(sentences, '\n')

    paragraphs = tf.ragged.constant([
        [['I', 'have', 'a', 'cat'], ['His', 'name', 'is', 'Mat']],
        [['Do', 'you', 'want', 'to', 'come', 'visit'], ["I'm", 'free', 'tomorrow']],
    ])
    print(paragraphs, '\n')

    logger.info('using tf.RaggedTensor.from_value_rowids():')
    print(tf.RaggedTensor.from_value_rowids(
        values=[3, 1, 4, 1, 5, 9, 2],
        value_rowids=[0, 0, 0, 0, 2, 2, 3]), '\n'
    )
    if args.plot:
        plt.figure()
        img = tf.io.read_file('supplement/tf2_g0109_01.png')
        img = tf.image.decode_png(img)
        plt.imshow(img)
        plt.show(block=False)

    logger.info('using tf.RaggedTensor.from_row_lengths():')
    print(tf.RaggedTensor.from_row_lengths(
        values=[3, 1, 4, 1, 5, 9, 2],
        row_lengths=[4, 0, 2, 1])
    )
    if args.plot:
        plt.figure()
        img = tf.io.read_file('supplement/tf2_g0109_02.png')
        img = tf.image.decode_png(img)
        plt.imshow(img)
        plt.show(block=False)

    logger.info('using tf.RaggedTensor.from_row_splits():')
    print(tf.RaggedTensor.from_row_splits(
        values=[3, 1, 4, 1, 5, 9, 2],
        row_splits=[0, 4, 4, 6, 7])
    )
    if args.plot:
        plt.figure()
        img = tf.io.read_file('supplement/tf2_g0109_03.png')
        img = tf.image.decode_png(img)
        plt.imshow(img)
        plt.show(block=False)
    

args.step = auto_increment(args.step, args.all)
### Step #4 - Overview: What you can store in a ragged tensor
if args.step == 4:
    print("\n### Step #4 - Overview: What you can store in a ragged tensor")

    logger.info('type=string, rank=2:')
    print(tf.ragged.constant([["Hi"], ["How", "are", "you"]]), '\n')  

    logger.info('type=int32, rank=3:')
    print(tf.ragged.constant([[[1, 2], [3]], [[4, 5]]]), '\n')

    logger.info('bad: multiple types:')
    try:
        tf.ragged.constant([["one", "two"], [3, 4]]) 
    except ValueError as exception:
        print(exception)

args.step = auto_increment(args.step, args.all)
### Step #5 - Example use case
if args.step == 5:
    print("\n### Step #5 - Example use case")
    
    print("\n\tMore to come!")


### End of File
print()
if args.plot:
    plt.show()
debug()
