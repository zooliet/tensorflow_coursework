#!/usr/bin/env python

# pip install -q git+https://github.com/tensorflow/docs

import sys
sys.path.append('./')
sys.path.append('../../')

from lab_utils import (
    tf, os, np, plt, logger, ap, BooleanAction,
    debug, toc, auto_increment
)

ap.add_argument('--epochs', type=int, default=1000, help='number of epochs: 1000*')
ap.add_argument('--batch', type=int, default=500, help='batch size: 500*')
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
import pathlib
import shutil
import tempfile

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - The Higgs Dataset
if args.step >= 1: 
    print("\n### Step #1 - The Higgs Dataset")

    # logdir
    logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
    shutil.rmtree(logdir, ignore_errors=True)

    url = 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz'
    dataset = f'{os.getenv("HOME")}/.keras/datasets/HIGGS.csv.gz'
    if not os.path.exists(dataset):
        dataset = tf.keras.utils.get_file('HIGGS.csv.gz', url)

    FEATURES = 28
    ds = tf.data.experimental.CsvDataset(
        dataset,
        [float(),]*(FEATURES+1), 
        compression_type="GZIP"
    )

    def pack_row(*row):
        label = row[0]
        features = tf.stack(row[1:], 1)
        return features, label
    # row = (None,),(None,),(None,)...(None,) # 29개 
    # *row = [(None,),(None,),(None,)...(None,)] # 29개 
    # row[0] = (None,) # 1개
    # row[1:] = [(None,),(None,)...(None,)] # 28개
    # tf.stack(row[1:], axis=1) => (None, 28)  
    # tf.stack(row[1:], axis=0) => (28, None)  

    packed_ds = ds.batch(10000).map(pack_row).unbatch()

    if args.step == 1:
        for features, label in packed_ds.batch(1000).take(1):
            logger.info(f'sample features:\n{features[0]}')
            logger.info(f'sample label: {label[0]}')

        if args.plot:
            plt.figure()
            plt.hist(features.numpy().flatten(), bins = 101)
            plt.show(block=False)

    # To keep this tutorial relatively short use just the first 1000 samples for validation, 
    # and the next 10 000 for training
    N_VALIDATION = int(1e3)
    N_TRAIN = int(1e4)
    BUFFER_SIZE = int(1e4)
    
    BATCH_SIZE = args.batch # 500
    STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE # 10000/500 = 20

    validate_ds = packed_ds.take(N_VALIDATION).cache()
    train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()

    train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)
    validate_ds = validate_ds.batch(BATCH_SIZE)

    # AUTOTUNE = tf.data.AUTOTUNE
    # # train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    # train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    # validate_ds = validate_ds.cache().prefetch(buffer_size=AUTOTUNE)


args.step = auto_increment(args.step, args.all)
### Step #2 - Demonstrate overfitting: Training procedure
if args.step >= 2:
    print("\n### Step #2 - Demonstrate overfitting: Training procedure")

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        0.001,
        decay_steps=STEPS_PER_EPOCH*1000,
        decay_rate=1,
        staircase=False
    )

    def get_optimizer():
        return tf.keras.optimizers.Adam(lr_schedule)

    if args.step == 2 and args.plot:
        step = np.linspace(0, 100000) # (50,), step == batch step
        lr = lr_schedule(step) # batch step별로 learning rate 계산 
        plt.figure(figsize = (8, 6))
        plt.plot(step/STEPS_PER_EPOCH, lr) # epoch별 leraning rate plot
        plt.ylim([0, max(plt.ylim())])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.show(block=False)

    def get_callbacks(name):
        return [
            tfdocs.modeling.EpochDots(),
            tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
            tf.keras.callbacks.TensorBoard(logdir/name),
        ]

    def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
        if optimizer is None:
            optimizer = get_optimizer()
            
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[
                tf.keras.losses.BinaryCrossentropy(from_logits=True, name='binary_crossentropy'),
                'accuracy'
            ]
        )
        # model.summary()

        history = model.fit(
            train_ds,
            steps_per_epoch = STEPS_PER_EPOCH,
            epochs=max_epochs, 
            validation_data=validate_ds,
            callbacks=get_callbacks(name),
            verbose=0
        )
        return history

    size_histories = {}


args.step = auto_increment(args.step, args.all)
### Step #3 - Demonstrate overfitting: Tiny model
if args.step in [3, 7, 8, 9, 10, 11, 12, 13]:
    print("\n### Step #3 - Demonstrate overfitting: Tiny model")

    tiny_model = Sequential([
        Dense(16, activation='elu', input_shape=(FEATURES,)),
        Dense(1)
    ])

    size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny', max_epochs=args.epochs)

    if args.step == 3 and args.plot:
        plt.figure()
        plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)
        plotter.plot(size_histories)
        plt.ylim([0.5, 0.7])
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #4 - Demonstrate overfitting: Small model
if args.step in [4, 7, 8]:
    print("\n### Step #4 - Demonstrate overfitting: Small model")

    small_model = Sequential([
        # `input_shape` is only required here so that `.summary` works.
        Dense(16, activation='elu', input_shape=(FEATURES,)),
        Dense(16, activation='elu'),
        Dense(1)
    ])

    size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small', max_epochs=args.epochs)


args.step = auto_increment(args.step, args.all)
### Step #5 - Demonstrate overfitting: Medium model
if args.step in [5, 7, 8]:
    print("\n### Step #5 - Demonstrate overfitting: Medium model")
    
    medium_model = Sequential([
        Dense(64, activation='elu', input_shape=(FEATURES,)),
        Dense(64, activation='elu'),
        Dense(64, activation='elu'),
        Dense(1)
    ])

    size_histories['Medium'] = compile_and_fit(medium_model, 'sizes/Medium', max_epochs=args.epochs)


args.step = auto_increment(args.step, args.all)
### Step #6 - Demonstrate overfitting: Large model
if args.step in [6, 7, 8]:
    print("\n### Step #6 - Demonstrate overfitting: Large model")
    
    large_model = Sequential([
        Dense(64, activation='elu', input_shape=(FEATURES,)),
        Dense(64, activation='elu'),
        Dense(64, activation='elu'),
        Dense(1)
    ])

    size_histories['Large'] = compile_and_fit(large_model, 'sizes/Large', max_epochs=args.epochs)


args.step = auto_increment(args.step, args.all)
### Step #7 - Demonstrate overfitting: Plot the training and validation losses
if args.step == 7:
    print("\n### Step #7 - Demonstrate overfitting: Plot the training and validation losses")

    if args.plot:
        plt.figure()
        plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)
        plotter.plot(size_histories)
        a = plt.xscale('log')
        plt.xlim([5, max(plt.xlim())])
        plt.ylim([0.5, 0.7])
        plt.xlabel("Epochs [Log Scale]")
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #8 - Demonstrate overfitting: View in TensorBoard
if args.step == 8:
    print("\n### Step #8 - Demonstrate overfitting: View in TensorBoard")

    logger.info(f'tensorboard --logdir {logdir}/sizes --bind_all')
    logger.info(f'tensorboard dev upload --logdir {logdir}/sizes')


args.step = auto_increment(args.step, args.all)
### Step #9 - Strategies to prevent overfitting
if args.step >= 9:
    print("\n### Step #9 - Strategies to prevent overfitting")

    shutil.rmtree(logdir/'regularizers/Tiny', ignore_errors=True)
    shutil.copytree(logdir/'sizes/Tiny', logdir/'regularizers/Tiny')
    regularizer_histories = {}
    regularizer_histories['Tiny'] = size_histories['Tiny']


args.step = auto_increment(args.step, args.all)
### Step #10 - Strategies to prevent overfitting: Add weight regularization
if args.step >= 10:
    print("\n### Step #10 - Strategies to prevent overfitting: Add weight regularization")

    l2_model = Sequential([
        Dense(
            512, activation='elu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            input_shape=(FEATURES,)
        ),
        Dense(
            512, activation='elu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ),
        Dense(
            512, activation='elu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ),
        Dense(
            512, activation='elu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ),
        Dense(1)
    ])

    regularizer_histories['l2'] = compile_and_fit(l2_model, "regularizers/l2")

    if args.step == 10 and args.plot:
        plt.figure()
        plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)
        plotter.plot(regularizer_histories)
        plt.ylim([0.5, 0.7])
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #11 - Strategies to prevent overfitting: Add dropout
if args.step >= 11:
    print("\n### Step #11 - Strategies to prevent overfitting: Add dropout")

    dropout_model = Sequential([
        Dense(512, activation='elu', input_shape=(FEATURES,)),
        Dropout(0.5),
        Dense(512, activation='elu'),
        Dropout(0.5),
        Dense(512, activation='elu'),
        Dropout(0.5),
        Dense(512, activation='elu'),
        Dropout(0.5),
        Dense(1)
    ])

    regularizer_histories['dropout'] = compile_and_fit(dropout_model, "regularizers/dropout")

    if args.step == 11 and args.plot:
            plt.figure()
            plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)
            plotter.plot(regularizer_histories)
            plt.ylim([0.5, 0.7])
            plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #12 - Strategies to prevent overfitting: Combined L2 + dropout
if args.step == 12:
    print("\n### Step #12 - Strategies to prevent overfitting: Combined L2 + dropout")

    combined_model = Sequential([
        Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='elu', input_shape=(FEATURES,)),
        Dropout(0.5),
        Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='elu'),
        Dropout(0.5),
        Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='elu'),
        Dropout(0.5),
        Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='elu'),
        Dropout(0.5),
        Dense(1)
    ])

    regularizer_histories['combined'] = compile_and_fit(combined_model, "regularizers/combined")

    if args.plot:
            plt.figure()
            plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)
            plotter.plot(regularizer_histories)
            plt.ylim([0.5, 0.7])
            plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #13 - Strategies to prevent overfitting: View in TensorBoard
if args.step == 13:
    print("\n### Step #13 - Strategies to prevent overfitting: View in TensorBoard")
    logger.info(f'tensorboard --logdir {logdir}/regularizers --bind_all')
    logger.info(f'tensorboard dev upload --logdir {logdir}/regularizers')


### End of File
if args.plot:
    plt.show()
debug()

