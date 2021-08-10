#!/usr/bin/env python

# pip install -U tfx

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
import urllib.request
import tempfile
import pandas as pd

# from tfx import v1 as tfx
import tfx
print('TFX version: {}'.format(tfx.__version__))

# from tensorflow.keras import Sequential, Model, Input
# from tensorflow.keras.layers import Flatten, Dense, Dropout, Softmax


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - Building a TFX Pipeline Locally
if args.step == 1:
    print("\n### Step #1 - Building a TFX Pipeline Locally")

    __doc__='''
    TFX makes it easier to orchestrate your machine learning (ML) workflow as a
    pipeline, in order to:
    - Automate your ML process, which lets you regularly retrain, evaluate, and
      deploy your model.
    - Create ML pipelines which include deep analysis of model performance and
      validation of newly trained models to ensure performance and reliability.
    - Monitor training data for anomalies and eliminate training-serving skew
    - Increase the velocity of experimentation by running a pipeline with
      different sets of hyperparameters.

    A typical pipeline development process begins on a local machine, with data
    analysis and component setup, before being deployed into production. This
    guide describes two ways to build a pipeline locally.
    - Customize a TFX pipeline template to fit the needs of your ML workflow.
      TFX pipeline templates are prebuilt workflows that demonstrate best
      practices using the TFX standard components.
    - Build a pipeline using TFX. In this use case, you define a pipeline
      without starting from a template.

    As you are developing your pipeline, you can run it with LocalDagRunner.
    Then, once the pipeline components have been well defined and tested, you
    would use a production-grade orchestrator such as Kubeflow or Airflow.
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #2 - Build a pipeline using a template: Create a copy of the pipeline template
if args.step == 2:
    print("\n### Step #2 - Build a pipeline using a template: Create a copy of the pipeline template")


args.step = auto_increment(args.step, args.all)
### Step #3 - Build a pipeline using a template: Explore the pipeline template
if args.step == 3:
    print("\n### Step #3 - Build a pipeline using a template: Explore the pipeline template")


args.step = auto_increment(args.step, args.all)
### Step #4 - Build a pipeline using a template: Customize your pipeline
if args.step == 4:
    print("\n### Step #4 - Build a pipeline using a template: Customize your pipeline")


args.step = auto_increment(args.step, args.all)
### Step #5 - Build a custom pipeline
if args.step == 5:
    print("\n### Step #5 - Build a custom pipeline")


### End of File
print()
if args.plot:
    plt.show()
debug()
