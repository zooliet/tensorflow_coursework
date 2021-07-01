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
### Step #1 - Controlling gradient recording
if args.step == 1:
    print("\n### Step #1 - Controlling gradient recording")

args.step = auto_increment(args.step, args.all)
### Step #2 - Stop gradient
if args.step == 2:
    print("\n### Step #2 - Stop gradient")

args.step = auto_increment(args.step, args.all)
### Step #3 - Custom gradients
if args.step == 3:
    print("\n### Step #3 - Custom gradients")

args.step = auto_increment(args.step, args.all)
### Step #4 - Custom gradient: Custom gradients in SavedModel
if args.step == 4:
    print("\n### Step #4 - Custom gradients: Custom gradients in SavedModel")

args.step = auto_increment(args.step, args.all)
### Step #5 - Multiple tapes
if args.step == 5:
    print("\n### Step #5 - Multiple tapes")

args.step = auto_increment(args.step, args.all)
### Step #6 - Multiple tapes: Higher-order gradients
if args.step == 6:
    print("\n### Step #6 - Multiple tapes: Higher-order gradients")

args.step = auto_increment(args.step, args.all)
### Step #7 - Jacobians
if args.step == 7:
    print("\n### Step #7 - Jacobians")

args.step = auto_increment(args.step, args.all)
### Step #8 - Jacobians: Scalar source
if args.step == 8:
    print("\n### Step #8 - Jacobians: Scalar source")

args.step = auto_increment(args.step, args.all)
### Step #9 - Jacobians: Tensor source
if args.step == 9:
    print("\n### Step #9 - Jacobians: Tensor source")

args.step = auto_increment(args.step, args.all)
### Step #10 - Jacobians: Batch Jacobian
if args.step == 10:
    print("\n### Step #10 - Jacobians: Batch Jacobian")


### End of File
print()
if args.plot:
    plt.show()
debug()
