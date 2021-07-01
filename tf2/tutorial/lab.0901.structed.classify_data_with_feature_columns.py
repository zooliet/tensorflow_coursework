#!/usr/bin/env python

# pip install -q git+https://github.com/tensorflow/examples.git

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
from PIL import Image
import pathlib

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Softmax
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - Use Pandas to create a dataframe
if args.step >= 1: 
    print("\n### Step #1 - Use Pandas to create a dataframe")


args.step = auto_increment(args.step, args.all)
### Step #2 - Create target variable
if args.step >= 2: 
    print("\n### Step #2 - Create target variable")


args.step = auto_increment(args.step, args.all)
### Step #3 - Split the dataframe into train, validation, and test
if args.step >= 3: 
    print("\n### Step #3 - Split the dataframe into train, validation, and test")


args.step = auto_increment(args.step, args.all)
### Step #4 - Create an input pipeline using tf.data
if args.step >= 4: 
    print("\n### Step #4 - Create an input pipeline using tf.data")


args.step = auto_increment(args.step, args.all)
### Step #5 - Understand the input pipeline
if args.step >= 5: 
    print("\n### Step #5 - Understand the input pipeline")


args.step = auto_increment(args.step, args.all)
### Step #6 - Demonstrate several types of feature columns
if args.step >= 6: 
    print("\n### Step #6 - Demonstrate several types of feature columns")


args.step = auto_increment(args.step, args.all)
### Step #6 - Demonstrate several types of feature columns: Numeric columns
if args.step >= 6: 
    print("\n### Step #6 - Demonstrate several types of feature columns: Numeric columns")


args.step = auto_increment(args.step, args.all)
### Step #7 - Demonstrate several types of feature columns: Bucketized columns
if args.step >= 7: 
    print("\n### Step #7 - Demonstrate several types of feature columns: Bucketized columns")


args.step = auto_increment(args.step, args.all)
### Step #8 - Demonstrate several types of feature columns: Categorical columns
if args.step >= 8: 
    print("\n### Step #8 - Demonstrate several types of feature columns: Categorical columns")


args.step = auto_increment(args.step, args.all)
### Step #9 - Demonstrate several types of feature columns: Embedding columns
if args.step >= 9: 
    print("\n### Step #9 - Demonstrate several types of feature columns: Embedding columns")


args.step = auto_increment(args.step, args.all)
### Step #10 - Demonstrate several types of feature columns: Hashed feature columns
if args.step >= 10: 
    print("\n### Step #10 - Demonstrate several types of feature columns: Hashed feature columns")


args.step = auto_increment(args.step, args.all)
### Step #11 - Demonstrate several types of feature columns: Crossed feature columns
if args.step >= 11: 
    print("\n### Step #11 - Demonstrate several types of feature columns: Crossed feature columns")


args.step = auto_increment(args.step, args.all)
### Step #12 - Choose which columns to use
if args.step >= 12: 
    print("\n### Step #12 - Choose which columns to use")


args.step = auto_increment(args.step, args.all)
### Step #13 - Choose which columns to use: Create a feature layer
if args.step >= 13: 
    print("\n### Step #13 - Choose which columns to use: Create a feature layer")


args.step = auto_increment(args.step, args.all)
### Step #14 - Create, compile, and train the model
if args.step >= 14: 
    print("\n### Step #14 - Create, compile, and train the model")



print()
if args.plot:
    plt.show()
debug()


