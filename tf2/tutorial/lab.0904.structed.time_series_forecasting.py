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


### Step #1 - The weather dataset
if args.step >= 1: 
    print("\n### Step #1 - The weather dataset")


args.step = auto_increment(args.step, args.all)
### Step #2 - The weather dataset: Inspect and cleanup
if args.step >= 2: 
    print("\n### Step #2 - The weather dataset: Inspect and cleanup")


args.step = auto_increment(args.step, args.all)
### Step #3 - The weather dataset: Feature engineering
if args.step >= 3: 
    print("\n### Step #3 - The weather dataset: Feature engineering")


args.step = auto_increment(args.step, args.all)
### Step #4 - The weather dataset: Split the data
if args.step >= 4: 
    print("\n### Step #4 - The weather dataset: Split the data")


args.step = auto_increment(args.step, args.all)
### Step #5 - The weather dataset: Normalize the data
if args.step >= 5: 
    print("\n### Step #5 - The weather dataset: Normalize the data")


args.step = auto_increment(args.step, args.all)
### Step #6 - Data windowing: Indexes and offsets
if args.step >= 6: 
    print("\n### Step #6 - Data windowing: Indexes and offsets")


args.step = auto_increment(args.step, args.all)
### Step #7 - Data windowing: Split
if args.step >= 7: 
    print("\n### Step #7 - Data windowing: Split")


args.step = auto_increment(args.step, args.all)
### Step #8 - Data windowing: Plot
if args.step >= 8: 
    print("\n### Step #8 - Data windowing: Plot")


args.step = auto_increment(args.step, args.all)
### Step #9 - Data windowing: Create tf.data.Datasets
if args.step >= 9: 
    print("\n### Step #9 - Data windowing: Create tf.data.Datasets")


args.step = auto_increment(args.step, args.all)
### Step #10 - Single step models
if args.step >= 10: 
    print("\n### Step #10 - Single step models")


args.step = auto_increment(args.step, args.all)
### Step #11 - Single step models: Baseline
if args.step >= 11: 
    print("\n### Step #11 - Single step models: Baseline")


args.step = auto_increment(args.step, args.all)
### Step #12 - Single step models: Linear model
if args.step >= 12: 
    print("\n### Step #12 - Single step models: Linear model")


args.step = auto_increment(args.step, args.all)
### Step #13 - Single step models: Dense
if args.step >= 13: 
    print("\n### Step #13 - Single step models: Dense")


args.step = auto_increment(args.step, args.all)
### Step #14 - Single step models: Multi-step dense
if args.step >= 14: 
    print("\n### Step #14 - Single step models: Multi-step dense")


args.step = auto_increment(args.step, args.all)
### Step #15 - Single step models: Convolution neural network
if args.step >= 15: 
    print("\n### Step #15 - Single step models: Convolution neural network")


args.step = auto_increment(args.step, args.all)
### Step #16 - Single step models: Recurrent neural network
if args.step >= 16: 
    print("\n### Step #16 - Single step models: Recurrent neural network")


args.step = auto_increment(args.step, args.all)
### Step #17 - Single step models: Performance
if args.step >= 17: 
    print("\n### Step #17 - Single step models: Performance")


args.step = auto_increment(args.step, args.all)
### Step #18 - Single step models: Multi-output models
if args.step >= 18: 
    print("\n### Step #18 - Single step models: Multi-output models")


args.step = auto_increment(args.step, args.all)
### Step #18 - Multi-step models
if args.step >= 18: 
    print("\n### Step #18 - Multi-step models")


args.step = auto_increment(args.step, args.all)
### Step #19 - Multi-step models: Baselines
if args.step >= 19: 
    print("\n### Step #19 - Multi-step models: Baselines")


args.step = auto_increment(args.step, args.all)
### Step #20 - Multi-step models: Single-shot models
if args.step >= 20: 
    print("\n### Step #20 - Multi-step models: Single-shot models")


args.step = auto_increment(args.step, args.all)
### Step #21 - Multi-step models: Autoregressive model
if args.step >= 21: 
    print("\n### Step #21 - Multi-step models: Autoregressive model")


args.step = auto_increment(args.step, args.all)
### Step #22 - Multi-step models: Performance
if args.step >= 22: 
    print("\n### Step #22 - Multi-step models: Performance")


args.step = auto_increment(args.step, args.all)
### Step #23 - Next steps
if args.step >= 23: 
    print("\n### Step #23 - Next steps")



### End of File
print()
if args.plot:
    plt.show()
debug()


