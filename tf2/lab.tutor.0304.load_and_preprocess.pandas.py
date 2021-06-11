#!/usr/bin/env python

import sys
sys.path.append('./')
sys.path.append('../')

from lab_utils import (
    os, np, plt, logger, ap, BooleanAction,
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
import pandas as pd

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("\n#################################################")
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Concatenate


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - Read data using pandas
if args.step >= 1: 
    print("\n### Step #1 - Read data using pandas")

    csv_file = tf.keras.utils.get_file(
        'heart.csv', 
        'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv'
    )

    df = pd.read_csv(csv_file)

    if args.step == 1:
        logger.info(f'{csv_file}:')
        print(df.head())

    # string to number
    df['thal'] = pd.Categorical(df['thal'])
    df['thal'] = df.thal.cat.codes

    if args.step == 1:
        print(df.head())
    

args.step = auto_increment(args.step, args.all)
### Step #2 - Load data using tf.data.Dataset
if args.step in [2, 3, 4]: 
    print("\n### Step #2 - Load data using tf.data.Dataset")

    target = df.pop('target')
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

    if args.step == 2:
        logger.info('dataset.element_spec:')
        print(*list(dataset.element_spec), sep='\n')
        print('')

        logger.info('dataset samples:')
        for feat, targ in dataset.take(5):
            print ('{} => {}'.format(feat, targ))

    train_dataset = dataset.shuffle(len(df)).batch(1)

    
args.step = auto_increment(args.step, args.all)
### Step #3 - Create and train a model
if args.step == 3: 
    print("\n### Step #3 - Create and train a model")

    def get_compiled_model():
        model = Sequential([
            Dense(10, activation='relu'),
            Dense(10, activation='relu'),
            Dense(1)
        ])

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        return model

    model = get_compiled_model()
    model.fit(train_dataset, epochs=args.epochs, verbose=2)


args.step = auto_increment(args.step, args.all)
### Step #4 - Alternative to feature columns
if args.step == 4:
    print("\n### Step #4 - Alternative to feature columns")

    inputs = {
        key: Input(shape=(1,), name=key) for key in df.keys()
    }

    logger.info('inputs:')
    for value in inputs.values():
        print(f'{value.name:8s} {value.shape} {value.dtype}')
    print('')
        
    # x = tf.stack(list(inputs.values()), axis=-1) # (None, 13, 1)
    # x = tf.reshape(x, (-1, len(inputs))) # (None, 13)
    # or
    x = Concatenate()(list(inputs.values())) # (None, 13)
    logger.info(f'Concatenate()(inputs.values()): {x.shape}\n')

    x = Dense(10, activation='relu')(x)
    output = Dense(1)(x)

    model_func = Model(inputs=inputs, outputs=output)

    model_func.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # input data
    dict_slices = tf.data.Dataset.from_tensor_slices(
        (df.to_dict('list'), target.values)
    ).batch(16)

    for feats, labels in dict_slices.take(1):
        logger.info('1 batch of dict_slices:')
        for key, value in feats.items():
            print(f'{key:10}:{value}')
        print("{:10s}:{}".format('labels', labels.numpy()))

    model_func.fit(dict_slices, epochs=args.epochs, verbose=0)


### End of File
if args.plot:
    plt.show()
debug()

