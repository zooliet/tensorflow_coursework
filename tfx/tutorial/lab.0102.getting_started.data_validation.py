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

from tfx import v1 as tfx
print('TFX version: {}'.format(tfx.__version__))

# from tensorflow.keras import Sequential, Model, Input
# from tensorflow.keras.layers import Flatten, Dense, Dropout, Softmax


### TOC
if args.step == 0:
    toc(__file__)


if args.step or args.all:
    if not os.path.exists('tmp/tfx_t0102/'):
        os.mkdir('tmp/tfx_t0102/') 


args.step = auto_increment(args.step, args.all)
### Step #1 - Setup: Set up variables
if args.step >= 1:
    print("\n### Step #1 - Setup: Set up variables")

    # We will create two pipelines. One for schema generation and one for training.
    SCHEMA_PIPELINE_NAME = "penguin-tfdv-schema"
    PIPELINE_NAME = "penguin-tfdv"

    # Output directory to store artifacts generated from the pipeline.
    SCHEMA_PIPELINE_ROOT = os.path.join('tmp/tfx_t0102/pipelines', SCHEMA_PIPELINE_NAME)
    PIPELINE_ROOT = os.path.join('tmp/tfx_t0102/pipelines', PIPELINE_NAME)

    # Path to a SQLite DB file to use as an MLMD storage.
    SCHEMA_METADATA_PATH = os.path.join('tmp/tfx_t0102/metadata', SCHEMA_PIPELINE_NAME, 'metadata.db')
    METADATA_PATH = os.path.join('tmp/tfx_t0102/metadata', PIPELINE_NAME, 'metadata.db')

    # Output directory where created models from the pipeline will be exported.
    SERVING_MODEL_DIR = os.path.join('tmp/tfx_t0102/serving_model', PIPELINE_NAME)

    # from absl import logging
    # logging.set_verbosity(logging.INFO)  # Set default logging level.


args.step = auto_increment(args.step, args.all)
### Step #2 - Setup: Prepare example data
if args.step >= 2:
    print("\n### Step #2 - Setup: Prepare example data")

    _data_url = 'https://raw.githubusercontent.com/tensorflow/tfx/master/tfx/examples/penguin/data/labelled/penguins_processed.csv'
    _data_filepath = tf.keras.utils.get_file(
        fname='data.csv',
        origin=_data_url, 
        cache_subdir='datasets/tfx-data'
    )
    DATA_ROOT = os.path.dirname(_data_filepath)

    if args.step == 2:
        logger.info(f'DATA_ROOT: {DATA_ROOT}')
        df = pd.read_csv(_data_filepath)
        print(df.head())


args.step = auto_increment(args.step, args.all)
### Step #3 - Generate a preliminary schema: Write a pipeline definition
if args.step >= 3:
    print("\n### Step #3 - Generate a preliminary schema: Write a pipeline definition")

    def _create_schema_pipeline(pipeline_name: str,
                                pipeline_root: str,
                                data_root: str,
                                metadata_path: str) -> tfx.dsl.Pipeline:

        """Creates a pipeline for schema generation."""
        # Brings data into the pipeline.
        example_gen = tfx.components.CsvExampleGen(input_base=data_root)

        # NEW: Computes statistics over data for visualization and schema generation.
        statistics_gen = tfx.components.StatisticsGen(
            examples=example_gen.outputs['examples']
        )

        # NEW: Generates schema based on the generated statistics.
        schema_gen = tfx.components.SchemaGen(
            statistics=statistics_gen.outputs['statistics'], 
            infer_feature_shape=True
        )

        components = [
            example_gen,
            statistics_gen,
            schema_gen,
        ]

        return tfx.dsl.Pipeline(
            pipeline_name=pipeline_name,
            pipeline_root=pipeline_root,
            metadata_connection_config=
                tfx.orchestration.metadata.sqlite_metadata_connection_config(metadata_path),
            components=components
        )


args.step = auto_increment(args.step, args.all)
### Step #4 - Generate a preliminary schema: Run the pipeline
if args.step >= 4:
    print("\n### Step #4 - Generate a preliminary schema: Run the pipeline")

    tfx.orchestration.LocalDagRunner().run(
        _create_schema_pipeline(
        pipeline_name=SCHEMA_PIPELINE_NAME,
        pipeline_root=SCHEMA_PIPELINE_ROOT,
        data_root=DATA_ROOT,
        metadata_path=SCHEMA_METADATA_PATH)
    )


args.step = auto_increment(args.step, args.all)
### Step #5 - Generate a preliminary schema: Review outputs of the pipeline
if args.step >= 5:
    print("\n### Step #5 - Generate a preliminary schema: Review outputs of the pipeline")

    from ml_metadata.proto import metadata_store_pb2
    # Non-public APIs, just for showcase.
    from tfx.orchestration.portable.mlmd import execution_lib

    # TODO(b/171447278): Move these functions into the TFX library.

    def get_latest_artifacts(metadata, pipeline_name, component_id):
        """Output artifacts of the latest run of the component."""
        context = metadata.store.get_context_by_type_and_name(
            'node', f'{pipeline_name}.{component_id}'
        )
        executions = metadata.store.get_executions_by_context(context.id)
        latest_execution = max(executions, key=lambda e:e.last_update_time_since_epoch)

        return execution_lib.get_artifacts_dict(
            metadata, latest_execution.id, metadata_store_pb2.Event.OUTPUT
        )

    # Non-public APIs, just for showcase.
    from tfx.orchestration.experimental.interactive import visualizations

    def visualize_artifacts(artifacts):
        """Visualizes artifacts using standard visualization modules."""
        for artifact in artifacts:
            visualization = visualizations.get_registry().get_visualization(artifact.type_name)
            if visualization:
                visualization.display(artifact)

    from tfx.orchestration.experimental.interactive import standard_visualizations
    standard_visualizations.register_standard_visualizations()

    # Non-public APIs, just for showcase.
    from tfx.orchestration.metadata import Metadata
    from tfx.types import standard_component_specs

    metadata_connection_config = tfx.orchestration.metadata.sqlite_metadata_connection_config(
        SCHEMA_METADATA_PATH
    )

    with Metadata(metadata_connection_config) as metadata_handler:
        # Find output artifacts from MLMD.
        stat_gen_output = get_latest_artifacts(metadata_handler, SCHEMA_PIPELINE_NAME, 'StatisticsGen')
        stats_artifacts = stat_gen_output[standard_component_specs.STATISTICS_KEY]
        schema_gen_output = get_latest_artifacts(metadata_handler, SCHEMA_PIPELINE_NAME, 'SchemaGen')
        schema_artifacts = schema_gen_output[standard_component_specs.SCHEMA_KEY]



    # debug()

    visualize_artifacts(stats_artifacts)
    visualize_artifacts(schema_artifacts)








# args.step = auto_increment(args.step, args.all)
# ### Step #3 - Create a pipeline: Write model training code
# if args.step >= 3:
#     print("\n### Step #3 - Create a pipeline: Write model training code")
#
#     _trainer_module_file = 'tfx/tutorial/labs/0101_penguin_trainer.py'
#     # logger.info(f'trainer module: {_trainer_module_file}')
#
#
# args.step = auto_increment(args.step, args.all)
# ### Step #4 - Create a pipeline: Write a pipeline definition
# if args.step >= 4:
#     print("\n### Step #4 - Create a pipeline: Write a pipeline definition")
#
#     __doc__='''
#     A Pipeline object represents a TFX pipeline which can be run using one of
#     pipeline orchestration systems that TFX supports.
#     '''
#     if args.step == 4: print(__doc__)
#
#     def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
#                         module_file: str, serving_model_dir: str,
#                         metadata_path: str) -> tfx.dsl.Pipeline:
#
#         """Creates a three component penguin pipeline with TFX."""
#
#         # Brings data into the pipeline.
#         example_gen = tfx.components.CsvExampleGen(input_base=data_root)
#
#         # Uses user-provided Python function that trains a model.
#         trainer = tfx.components.Trainer(
#             module_file=module_file,
#             examples=example_gen.outputs['examples'],
#             train_args=tfx.proto.TrainArgs(num_steps=100),
#             eval_args=tfx.proto.EvalArgs(num_steps=5),
#             custom_config={'debug': True if args.debug else False}
#         )
#
#         # Pushes the model to a filesystem destination.
#         pusher = tfx.components.Pusher(
#             model=trainer.outputs['model'],
#             push_destination=tfx.proto.PushDestination(
#                 filesystem=tfx.proto.PushDestination.Filesystem(
#                     base_directory=serving_model_dir
#                 )
#             )
#         )
#
#         # Following three components will be included in the pipeline.
#         components = [
#             example_gen,
#             trainer,
#             pusher,
#         ]
#
#         return tfx.dsl.Pipeline(
#             pipeline_name=pipeline_name,
#             pipeline_root=pipeline_root,
#             metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(metadata_path),
#             components=components
#         )
#
#
# args.step = auto_increment(args.step, args.all)
# ### Step #5 - Run the pipeline
# if args.step == 5:
#     print("\n### Step #5 - Run the pipeline")
#
#     __doc__='''
#     TFX supports multiple orchestrators to run pipelines. In this tutorial we
#     will use LocalDagRunner which is included in the TFX Python package and
#     runs pipelines on local environment. We often call TFX pipelines "DAGs"
#     which stands for directed acyclic graph.
#
#     LocalDagRunner provides fast iterations for developemnt and debugging. TFX
#     also supports other orchestrators including Kubeflow Pipelines and Apache
#     Airflow which are suitable for production use cases.
#     '''
#     print(__doc__)
#
#     logger.info(f'pipeline_name: {PIPELINE_NAME}')
#     logger.info(f'pipeline_root: {PIPELINE_ROOT}')
#     logger.info(f'data_root: {DATA_ROOT}')
#     logger.info(f'module_file: {_trainer_module_file}')
#     logger.info(f'serving_model_dir: {SERVING_MODEL_DIR}')
#     logger.info(f'metadata_path: {METADATA_PATH}')
#
#     tfx.orchestration.LocalDagRunner().run(
#         _create_pipeline(
#             pipeline_name=PIPELINE_NAME,
#             pipeline_root=PIPELINE_ROOT,
#             data_root=DATA_ROOT,
#             module_file=_trainer_module_file,
#             serving_model_dir=SERVING_MODEL_DIR,
#             metadata_path=METADATA_PATH
#         )
#     )

### End of File
print()
if args.plot:
    plt.show()
debug()
