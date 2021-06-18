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


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - Creating a SavedModel from Keras
if args.step == 1:
    print("\n### Step #1 - Creating a SavedModel from Keras")


args.step = auto_increment(args.step, args.all)
### Step #2 - Running a SavedModel in TensorFlow Serving
if args.step == 2:
    print("\n### Step #2 - Running a SavedModel in TensorFlow Serving")


args.step = auto_increment(args.step, args.all)
### Step #3 - The SavedModel format on disk
if args.step == 3:
    print("\n### Step #3 - The SavedModel format on disk")


args.step = auto_increment(args.step, args.all)
### Step #4 - Saving a custom model
if args.step == 4:
    print("\n### Step #4 - Saving a custom model")


args.step = auto_increment(args.step, args.all)
### Step #5 - Loading and using a custom model
if args.step == 5:
    print("\n### Step #5 - Loading and using a custom model")


args.step = auto_increment(args.step, args.all)
### Step #6 - Loading and using a custom model: Basic fine-tuning
if args.step == 6:
    print("\n### Step #6 - Loading and using a custom model: Basic fine-tuning")


args.step = auto_increment(args.step, args.all)
### Step #7 - Loading and using a custom model: General fine-tuning
if args.step == 7:
    print("\n### Step #7 - Loading and using a custom model: General fine-tuning")


args.step = auto_increment(args.step, args.all)
### Step #8 - Specifying signatures during export
if args.step == 8:
    print("\n### Step #8 - Specifying signatures during export")


args.step = auto_increment(args.step, args.all)
### Step #9 - Load a SavedModel in C++
if args.step == 9:
    print("\n### Step #9 - Load a SavedModel in C++")


args.step = auto_increment(args.step, args.all)
### Step #10 - Details of the SavedModel command line interface: Install the SavedModel CLI
if args.step == 10:
    print("\n### Step #10 - Details of the SavedModel command line interface: Install the SavedModel CLI")


args.step = auto_increment(args.step, args.all)
### Step #11 - Details of the SavedModel command line interface: Overview of commands
if args.step == 11:
    print("\n### Step #11 - Details of the SavedModel command line interface: Overview of commands")


args.step = auto_increment(args.step, args.all)
### Step #12 - Details of the SavedModel command line interface: show command
if args.step == 12:
    print("\n### Step #12 - Details of the SavedModel command line interface: show command")


args.step = auto_increment(args.step, args.all)
### Step #13 - Details of the SavedModel command line interface: run command
if args.step == 13:
    print("\n### Step #13 - Details of the SavedModel command line interface: run command")


### End of File
if args.plot:
    plt.show()
debug()

