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


# if args.step or args.all:
#     if not os.path.exists('tmp/tfx_g0101/'):
#         os.mkdir('tmp/tfx_g0101/')


args.step = auto_increment(args.step, args.all)
### Step #1 - Artifact
if args.step == 1:
    print("\n### Step #1 - Artifact")

    __doc__='''
    The outputs of steps in a TFX pipeline are called artifacts. Subsequent
    steps in your workflow may use these artifacts as inputs. In this way, TFX
    lets you transfer data between workflow steps.

    For instance, the ExampleGen standard component emits serialized examples,
    which components such as the StatisticsGen standard component use as
    inputs.

    Artifacts must be strongly typed with an artifact type registered in the ML
    Metadata store. 

    Artifact types have a name and define a schema of its properties. Artifact
    type names must be unique in your ML Metadata store. TFX provides several
    standard artifact types that describe complex data types and value types,
    such as: string, integer, and float. You can reuse these artifact types or
    define custom artifact types that derive from Artifact.
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #2 - Parameter
if args.step == 2:
    print("\n### Step #2 - Parameter")

    __doc__='''
    Parameters are inputs to pipelines that are known before your pipeline is
    executed. Parameters let you change the behavior of a pipeline, or a part
    of a pipeline, through configuration instead of code.

    For example, you can use parameters to run a pipeline with different sets
    of hyperparameters without changing the pipeline's code.

    Using parameters lets you increase the velocity of experimentation by
    making it easier to run your pipeline with different sets of parameters.
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #3 - Component
if args.step == 3:
    print("\n### Step #3 - Component")

    __doc__='''
    A component is an implementation of an ML task that you can use as a step
    in your TFX pipeline. Components are composed of:

    - A component specification, which defines the component's input and output
      artifacts, and the component's required parameters.  
    - An executor, which implements the code to perform a step in your ML
      workflow, such as ingesting and transforming data or training and
      evaluating a model.  
    - A component interface, which packages the component specification and
      executor for use in a pipeline.  

    TFX provides several standard components that you can use in your
    pipelines. If these components do not meet your needs, you can build custom
    components. 
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #4 - Pipeline
if args.step == 4:
    print("\n### Step #4 - Pipeline")

    __doc__='''
    A TFX pipeline is a implementation of an ML workflow that can be
    run on various orchestrators, such as: Apache Airflow, Apache Beam, and
    Kubeflow Pipelines. A pipeline is composed of component instances and input
    parameters.

    Component instances produce artifacts as outputs and typically depend on
    artifacts produced by upstream component instances as inputs. 

    For example, consider a pipeline that does the following:

    - Ingests data directly from a proprietary system using a custom component.
    - Calculates statistics for the training data using the StatisticsGen
      component.
    - Creates a data schema using the SchemaGen component.
    - Checks the training data for anomalies using the ExampleValidator
      component.
    - Performs feature engineering on the dataset using the Transform
      component.
    - Trains a model using the Trainer component.
    - Evaluates the trained model using the Evaluator component.
    - If the model passes its evaluation, the pipeline enqueues the trained
      model to a proprietary deployment system using a custom component.

    To determine the execution sequence for the component instances, TFX
    analyzes the artifact dependencies.

    - The data ingestion component does not have any artifact dependencies, so
      it can be the first node in the graph.
    - StatisticsGen depends on the examples produced by data ingestion, so it
      must be executed after data ingestion.
    - SchemaGen depends on the statistics created by StatisticsGen, so it must
      be executed after StatisticsGen.
    - ExampleValidator depends on the statistics created by StatisticsGen and
      the schema created by SchemaGen, so it must be executed after
      StatisticsGen and SchemaGen.
    - Transform depends on the examples produced by data ingestion and the
      schema created by SchemaGen, so it must be executed after data ingestion
      and SchemaGen.
    - Trainer depends on the examples produced by data ingestion, the schema
      created by SchemaGen, and the saved model produced by Transform. The
      Trainer can be executed only after data ingestion, SchemaGen, and
      Transform.
    - Evaluator depends on the examples produced by data ingestion and the
      saved model produced by the Trainer, so it must be executed after data
      ingestion and the Trainer.
    - The custom deployer depends on the saved model produced by the Trainer
      and the analysis results created by the Evaluator, so the deployer must
      be executed after the Trainer and the Evaluator.
      
    Based on this analysis, an orchestrator runs:

    - The data ingestion, StatisticsGen, SchemaGen component instances
      sequentially.
    - The ExampleValidator and Transform components can run in parallel since
      they share input artifact dependencies and do not depend on each other's
      output.
    - After the Transform component is complete, the Trainer, Evaluator, and
      custom deployer component instances run sequentially.
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #5 - TFX Pipeline Template
if args.step == 5:
    print("\n### Step #5 - TFX Pipeline Template")

    __doc__='''
    TFX Pipeline Templates make it easier to get started with pipeline
    development by providing a prebuilt pipeline that you can customize for
    your use case.
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #6 - Pipeline Run
if args.step == 6:
    print("\n### Step #6 - Pipeline Run")

    __doc__='''
    A run is a single execution of a pipeline.
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #7 - Orchestrator
if args.step == 7:
    print("\n### Step #7 - Orchestrator")

    __doc__='''
    An Orchestrator is a system where you can execute pipeline runs. 
    TFX supports orchestrators such as: Apache Airflow, Apache Beam, and
    Kubeflow Pipelines. TFX also uses the term DagRunner to refer to an
    implementation that supports an orchestrator.
    '''
    print(__doc__)


# args.step = auto_increment(args.step, args.all)
# ### Step #8 - Summary
# if args.step == 8:
#     print("\n### Step #8 - Summary")
#
#     __doc__='''
#     pipleline == workflow
#     a pipeline consists of components
#     components == tasks
#     '''
#     print(__doc__)

### End of File
print()
if args.plot:
    plt.show()
debug()
