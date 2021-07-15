#!/usr/bin/env python

import sys
sys.path.append('./')
sys.path.append('../../')

from lab_utils import (
    tf, os, np, plt, logger, ap, BooleanAction,
    debug, toc, auto_increment
)

ap.add_argument('--epochs', type=int, default=10, help='number of epochs: 10*')
ap.add_argument('--batch', type=int, default=32, help='batch size: 32*')
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

# if args.steps:
#     args.steps = [int(n) for n in args.steps.split(",")]
#     logger.info(f'Step #{args.steps} will be run')
# else:
#     args.steps = []


import time
import pandas as pd
import itertools
import pathlib
import re

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.layers.experimental import preprocessing

### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - Load the dataset 
if args.step >= 1:
    print("\n### Step #1 - Load the dataset")

    titanic = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
    logger.debug(f'\n{titanic.head()}')

    titanic_features = titanic.copy()
    titanic_labels = titanic_features.pop('survived')


### Step #2 - Construct multi-input layers
if args.step >= 2:
    print("\n### Step #2 - Construct input layers")

    print("\n### Step #2 - Construct input layers")
    inputs = {}
    for name, column in titanic_features.items():
        dtype = column.dtype
        if dtype == object:
            dtype = tf.string
        else:
            dtype = tf.float32
        inputs[name] = Input(shape=(1,), name=name, dtype=dtype)

    if args.verbose: print(*inputs.items(), sep='\n')


### Step #3 - Collect numeic columns
if args.step >= 3:
    print("\n### Step #3 - Collect numeic columns")

    numeric_inputs = dict(filter(lambda ele: ele[1].dtype == tf.float32, inputs.items()))
    # = numeric_inputs = {
    #     name: input for (name, input) in inputs.items() if input.dtype==tf.float32
    # }
    if args.verbose: print(*numeric_inputs.items(), sep='\n')


### Step #4 - Prepare normalization
if args.step >= 4:
    print("\n### Step #4 - Prepare normalization")

    norm = preprocessing.Normalization()

    titanic_numeric_features = titanic_features[numeric_inputs.keys()]
    # = titanic_numeric_features = titanic_features[['age', 'n_siblings_spouses', 'fare', 'parch']]

    norm.adapt(titanic_numeric_features)
    # = norm.adapt(np.array(titanic_numeric_features))
    # = norm.adapt(titanic_numeric_features.values)

    sample = titanic_numeric_features[0:2]
    normalized = norm(sample)
    logger.debug(f'norm(sample):\n{normalized}')


### Step #5 - Build the model: for numeric inputs   
if args.step >= 5:
    print("\n### Step #5 - Build the model: for numeric inputs")

    x = Concatenate()(numeric_inputs.values())
    # = inputs.values() == [input_age, input_siblings, input_fare, input_parch]
    # ? Concatenate()(list(numeric_inputs.values()))
    x_numeric = norm(x)

    x = Dense(64)(x_numeric)
    output = Dense(1)(x)

    model_numeric = Model(inputs=numeric_inputs, outputs=[output])

    model_numeric.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam()
    )
    if args.verbose:
        model_numeric.summary()


### Step #6 - Data preparation: for numeric inputs
if args.step in [6, 7]:
    print("\n### Step #6 - Data preparation: for numeric inputs")

    logger.info('1. primitively')
    data_primitive = {
        'age': titanic_features['age'].values,
        'n_siblings_spouses': titanic_features['n_siblings_spouses'].values,
        'fare': titanic_features['fare'].values,
        'parch': titanic_features['parch'].values,
    }

    logger.info('2. using dict()')
    for name, column in titanic_features.items():
        dtype = column.dtype
        if dtype == object:
            dtype = tf.string
            _ = titanic_features.pop(name)
        else:
            dtype = tf.float32
    # = titanic_features.pop('deck')
    # = titanic_features.pop('embark_town')
    # = titanic_features.pop('alone')
    # = titanic_features.pop('class')
    # = titanic_features.pop('sex')
    data_dict = dict(titanic_features)

    logger.info('3. dataset form')
    data_ds = tf.data.Dataset.from_tensor_slices(
        (dict(titanic_features), titanic_labels)
    )
    data_ds = data_ds.shuffle(len(titanic_labels)).batch(32)


### Step #7 - Train the numeric-only model
if args.step == 7:
    print("\n### Step #7 - Train the numeric-only model")

    logger.info('with data_primitive')
    model_numeric.fit(
        data_primitive,
        titanic_labels,
        # = titanic_lables.values,
        epochs=args.epochs,
        verbose=2 if args.verbose else 0
    )
    print()

    logger.info('with data_dict')
    model_numeric.fit(
        x=data_dict,
        y=titanic_labels,
        epochs=args.epochs,
        verbose=2 if args.verbose else 0
    )
    print()

    logger.info('with data_ds')
    model_numeric.fit(
        data_ds,
        epochs=args.epochs,
        verbose=2 if args.verbose else 0
    )


### Step #8 - Preprocessing layer: including categorical inputs
if args.step >= 8:
    print("\n### Step #8 - Preprocessing layer: including categorical inputs")

    preprocessed_inputs = [x_numeric] 
    # for categorical feature 
    for name, input in inputs.items():
        if input.dtype == tf.float32:
            continue

        # string to number
        lookup = preprocessing.StringLookup(vocabulary=np.unique(titanic[name]))
        # number to onehot
        one_hot = preprocessing.CategoryEncoding(num_tokens=lookup.vocabulary_size())

        x_categorical = lookup(input)
        x_categorical = one_hot(x_categorical)
        preprocessed_inputs.append(x_categorical)

    preprocessed_output = Concatenate()(preprocessed_inputs)

    preprocessing_model = Model(inputs, preprocessed_output)

    if args.verbose:
        preprocessing_model.summary()


### Step #9 - Build the model
if args.step >= 9:
    print("\n### Step #9 - Build the model")

    def titanic_model(preprocessing_head, inputs):
        body = Sequential([
            Dense(64),
            Dense(1)
        ])

        preprocessed_inputs = preprocessing_head(inputs)
        result = body(preprocessed_inputs)
        model = Model(inputs, result)

        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam()
        )
        return model

    # = x = preprocessing_model(inputs)
    # x = Dense(64)(x)
    # output = Dense(1)(x)
    #
    # model = Model(inputs, output)
    #
    # model.compile(
    #     loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #     optimizer=tf.keras.optimizers.Adam()
    # )

    model = titanic_model(preprocessing_model, inputs)
    if args.verbose:
        model.summary()


### Step #10 - Data preparation: for all inputs
if args.step in [10, 11]:
    print("\n### Step #10 - Data preparation: for all inputs")

    titanic_features = titanic.copy()
    titanic_labels = titanic_features.pop('survived')

    logger.info('1. using dict()')
    data_dict = dict(titanic_features)
    # = data_dict = {
    #     name: np.array(value) for name, value in titanic_features.items()
    # }

    logger.info('2. dataset form')
    data_ds = tf.data.Dataset.from_tensor_slices(
        (dict(titanic_features), titanic_labels)
    )
    data_ds = data_ds.shuffle(len(titanic_labels)).batch(32)
    

### Step #11 - Train the model: for all inputs
if args.step == 11:
    print("\n### Step #11 - Train the model: for all inputs")

    model = titanic_model(preprocessing_model, inputs)

    logger.info('with data_dict')
    model.fit(
        x=data_dict,
        y=titanic_labels,
        epochs=args.epochs,
        verbose=2 if args.verbose else 0
    )
    print()

    logger.info('with data_ds')
    model.fit(
        data_ds,
        epochs=args.epochs,
        verbose=2 if args.verbose else 0
    )


### Step #12 - Dataset from disk 
if args.step >= 12:
    print("\n### Step #12 - Dataset from disk")

    titanic_file_path = f'{os.getenv("HOME")}/.keras/datasets/train.csv'
    
    titanic_csv_ds = tf.data.experimental.make_csv_dataset(
        titanic_file_path,
        batch_size=32, 
        label_name='survived',
        num_epochs=1,
        ignore_errors=True,
    )

    if args.verbose:
        for batch, label in titanic_csv_ds.unbatch().batch(5).take(1):
            for key, value in batch.items():
                print(f"{key:20s}: {value}")
                print(f"{'label':20s}: {label}")
            print()


### Step #13 - Train the model 
if args.step >= 13:
    print("\n### Step #13 - Train the model")

    model = titanic_model(preprocessing_model, inputs)

    model.fit(
        titanic_csv_ds,
        epochs=args.epochs,
        verbose=2 if args.verbose else 0
    )


### End of File
print()
if args.plot:
    plt.show()
debug()
