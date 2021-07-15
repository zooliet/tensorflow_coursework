#!/usr/bin/env python

import sys
sys.path.append('./')
sys.path.append('../../')

from lab_utils import (
    tf, os, np, plt, logger, ap, BooleanAction,
    debug, toc, auto_increment
)

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
import pandas as pd

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.layers.experimental import preprocessing

### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - tf.stack of list: rank 1s to a rank 2  
if args.step == 1:
    print("\n### Step #1 - tf.stack of list: rank 1s to a rank 2")

    a = ['a1', 'a2', 'a3', 'a4']
    b = ['b1', 'b2', 'b3', 'b4']
    c = ['c1', 'c2', 'c3', 'c4']

    tf_a = tf.convert_to_tensor(a)
    tf_b = tf.convert_to_tensor(b)
    tf_c = tf.convert_to_tensor(c)

    tf_s1 = tf.stack([tf_a, tf_b, tf_c], axis=0)
    tf_s2 = tf.stack([tf_a, tf_b, tf_c], axis=1) # -1

    logger.debug(f'tf_s1{tf_s1.shape} vs tf_s2{tf_s2.shape}:\n{tf_s1}\n---------\n{tf_s2}')


args.step = auto_increment(args.step, args.all)
### Step #2 - tf.stack of vector: rank 2s to a rank 3  
if args.step == 2:
    print("\n### Step #2 - tf.stack of vector: rank 2s to a rank 3")

    a = [['a11', 'a12', 'a13', 'a14'], ['a21', 'a22', 'a23', 'a24']]
    b = [['b11', 'b12', 'b13', 'b14'], ['b21', 'b22', 'b23', 'b24']]
    c = [['c11', 'c12', 'c13', 'c14'], ['c21', 'c22', 'c23', 'c24']]

    tf_a = tf.convert_to_tensor(a)
    tf_b = tf.convert_to_tensor(b)
    tf_c = tf.convert_to_tensor(c)

    tf_s1 = tf.stack([tf_a, tf_b, tf_c], axis=0)
    tf_s2 = tf.stack([tf_a, tf_b, tf_c], axis=1) 
    tf_s3 = tf.stack([tf_a, tf_b, tf_c], axis=2) # -1 
   
    logger.debug(f'tf_s1{tf_s1.shape} vs tf_s2{tf_s2.shape} vs tf_s3{tf_s3.shape}:\
            \r{tf_s1}\n---------\n{tf_s2}\n---------\n{tf_s3}\n')

    # # numpy
    # np_a = np.array(a)
    # np_b = np.array(b)
    # np_c = np.array(c)
    #
    # np_s1 = np.stack([np_a, np_b, np_c], axis=0)
    # np_s2 = np.stack([np_a, np_b, np_c], axis=1)
    # np_s3 = np.stack([np_a, np_b, np_c], axis=2) # -1
    #
    # logger.debug(f'np_s1{np_s1.shape} vs np_s2{np_s2.shape} vs np_s3{np_s3.shape}:\
    #         \r{np_s1}\n---------\n{np_s2}\n---------\n{np_s3}')


args.step = auto_increment(args.step, args.all)
### Step #3 - tf.concat and np.concantenate  
if args.step == 3:
    print("\n### Step #3 - tf.concat and np.concatenate")

    a = [['a11', 'a12', 'a13', 'a14'], ['a21', 'a22', 'a23', 'a24']]
    b = [['b11', 'b12', 'b13', 'b14'], ['b21', 'b22', 'b23', 'b24']]
    c = [['c11', 'c12', 'c13', 'c14'], ['c21', 'c22', 'c23', 'c24']]

    tf_a = tf.convert_to_tensor(a)
    tf_b = tf.convert_to_tensor(b)
    tf_c = tf.convert_to_tensor(c)

    tf_s1 = tf.concat([tf_a, tf_b, tf_c], axis=0)
    tf_s2 = tf.concat([tf_a, tf_b, tf_c], axis=1) 
    
    logger.debug(f'tf_s1{tf_s1.shape} vs tf_s2{tf_s2.shape}:\n{tf_s1}\n---------\n{tf_s2}\n')

    # numpy
    np_a = np.array(a)
    np_b = np.array(b)
    np_c = np.array(c)

    np_s1 = np.concatenate([np_a, tf_b, tf_c], axis=0)
    np_s2 = np.concatenate([np_a, tf_b, tf_c], axis=1) 
    
    logger.debug(f'np_s1{np_s1.shape} vs np_s2{np_s2.shape}:\n{np_s1}\n---------\n{np_s2}')


args.step = auto_increment(args.step, args.all)
### Step #4 - np.vstack and np.hstack   
if args.step == 4:
    print("\n### Step #4 - np.vstack and np.hstack")

    a = [['a11', 'a12', 'a13', 'a14'], ['a21', 'a22', 'a23', 'a24']]
    b = [['b11', 'b12', 'b13', 'b14'], ['b21', 'b22', 'b23', 'b24']]
    c = [['c11', 'c12', 'c13', 'c14'], ['c21', 'c22', 'c23', 'c24']]

    np_a = np.array(a)
    np_b = np.array(b)
    np_c = np.array(c)

    np_s1 = np.vstack((np_a, np_b, np_c))
    np_s2 = np.hstack((np_a, np_b, np_c))
    
    logger.debug(f'np_s1{np_s1.shape} vs np_s2{np_s2.shape}:\n{np_s1}\n---------\n{np_s2}')


### End of File
print()
if args.plot:
    plt.show()
debug()
