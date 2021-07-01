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
### Step #1 - Data processing and exploration: Download the Kaggle Credit Card Fraud data set
if args.step >= 1: 
    print("\n### Step #1 - Data processing and exploration: Download the Kaggle Credit Card Fraud data set")


args.step = auto_increment(args.step, args.all)
### Step #2 - Data processing and exploration: Examine the class label imbalance
if args.step >= 2: 
    print("\n### Step #2 - Data processing and exploration: Examine the class label imbalance")


args.step = auto_increment(args.step, args.all)
### Step #3 - Data processing and exploration: Clean, split and normalize the data
if args.step >= 3: 
    print("\n### Step #3 - Data processing and exploration: Clean, split and normalize the data")


args.step = auto_increment(args.step, args.all)
### Step #4 - Data processing and exploration: Look at the data distribution
if args.step >= 4: 
    print("\n### Step #4 - Data processing and exploration: Look at the data distribution")


args.step = auto_increment(args.step, args.all)
### Step #5 - Define the model and metrics
if args.step >= 5: 
    print("\n### Step #5 - Define the model and metrics")


args.step = auto_increment(args.step, args.all)
### Step #6 - Define the model and metrics: Understanding useful metrics
if args.step >= 6: 
    print("\n### Step #6 - Define the model and metrics: Understanding useful metrics")


args.step = auto_increment(args.step, args.all)
### Step #7 - Baseline model: Build the model
if args.step >= 7: 
    print("\n### Step #7 - Baseline model: Build the model")


args.step = auto_increment(args.step, args.all)
### Step #8 - Baseline model: Set the correct initial bias
if args.step >= 8: 
    print("\n### Step #8 - Baseline model: Set the correct initial bias")


args.step = auto_increment(args.step, args.all)
### Step #9 - Baseline model: Checkpoint the initial weights
if args.step >= 9: 
    print("\n### Step #9 - Baseline model: Checkpoint the initial weights")


args.step = auto_increment(args.step, args.all)
### Step #10 - Baseline model: Confirm that the bias fix helps
if args.step >= 10: 
    print("\n### Step #10 - Baseline model: Confirm that the bias fix helps")


args.step = auto_increment(args.step, args.all)
### Step #11 - Baseline model: Train the model
if args.step >= 11: 
    print("\n### Step #11 - Baseline model: Train the model")


args.step = auto_increment(args.step, args.all)
### Step #12 - Baseline model: Check training history
if args.step >= 12: 
    print("\n### Step #12 - Baseline model: Check training history")


args.step = auto_increment(args.step, args.all)
### Step #13 - Baseline model: Evaluate metrics
if args.step >= 13: 
    print("\n### Step #13 - Baseline model: Evaluate metrics")


args.step = auto_increment(args.step, args.all)
### Step #13 - Baseline model: Plot the ROC
if args.step >= 13: 
    print("\n### Step #13 - Baseline model: Plot the ROC")


args.step = auto_increment(args.step, args.all)
### Step #14 - Baseline model: Plot the AUPRC
if args.step >= 14: 
    print("\n### Step #14 - Baseline model: Plot the AUPRC")


args.step = auto_increment(args.step, args.all)
### Step #15 - Class weights: Calculate class weights
if args.step >= 15: 
    print("\n### Step #15 - Class weights: Calculate class weights")


args.step = auto_increment(args.step, args.all)
### Step #16 - Class weights: Train a model with class weights
if args.step >= 16: 
    print("\n### Step #16 - Class weights: Train a model with class weights")


args.step = auto_increment(args.step, args.all)
### Step #17 - Class weights: Check training history
if args.step >= 17: 
    print("\n### Step #17 - Class weights: Check training history")


args.step = auto_increment(args.step, args.all)
### Step #18 - Class weights: Evaluate metrics
if args.step >= 18: 
    print("\n### Step #18 - Class weights: Evaluate metrics")


args.step = auto_increment(args.step, args.all)
### Step #19 - Class weights: Plot the ROC
if args.step >= 19: 
    print("\n### Step #19 - Class weights: Plot the ROC")


args.step = auto_increment(args.step, args.all)
### Step #20 - Class weights: Plot the AUPRC
if args.step >= 20: 
    print("\n### Step #20 - Class weights: Plot the AUPRC")


args.step = auto_increment(args.step, args.all)
### Step #21 - Oversampling: Oversample the minority class
if args.step >= 21: 
    print("\n### Step #21 - Oversampling: Oversample the minority class")


args.step = auto_increment(args.step, args.all)
### Step #22 - Oversampling: Train on the oversampled data
if args.step >= 22: 
    print("\n### Step #22 - Oversampling: Oversample the minority class: Train on the oversampled data")



args.step = auto_increment(args.step, args.all)
### Step #23 - Oversampling: Check training history
if args.step >= 23: 
    print("\n### Step #23 - Oversampling: Check training history")


args.step = auto_increment(args.step, args.all)
### Step #24 - Oversampling: Re-train
if args.step >= 24: 
    print("\n### Step #24 - Oversampling: Re-train")


args.step = auto_increment(args.step, args.all)
### Step #25 - Oversampling: Re-check training history
if args.step >= 25: 
    print("\n### Step #25 - Oversampling: Re-check training history")


args.step = auto_increment(args.step, args.all)
### Step #26 - Oversampling: Evaluate metrics
if args.step >= 26: 
    print("\n### Step #26 - Oversampling: Evaluate metrics")


args.step = auto_increment(args.step, args.all)
### Step #27 - Oversampling: Plot the ROC
if args.step >= 27: 
    print("\n### Step #27 - Oversampling: Plot the ROC")


args.step = auto_increment(args.step, args.all)
### Step #28 - Oversampling: Plot the AUPRC
if args.step >= 28: 
    print("\n### Step #28 - Oversampling: Plot the AUPRC")


args.step = auto_increment(args.step, args.all)
### Step #29 - Applying this tutorial to your problem
if args.step >= 29: 
    print("\n### Step #29 - Applying this tutorial to your problem")




### End of File
print()
if args.plot:
    plt.show()
debug()


