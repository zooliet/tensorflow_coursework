#!/usr/bin/env python

# pip install -q seaborn

import sys
sys.path.append('./')
sys.path.append('../../')

from lab_utils import (
    tf, os, np, plt, logger, ap, BooleanAction,
    debug, toc, auto_increment
)

ap.add_argument('--epochs', type=int, default=100, help='number of epochs: 100*')
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

import time
import pandas as pd
import seaborn as sns

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers.experimental.preprocessing import Normalization


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - The Auto MPG dataset: Get the data
if args.step >= 1: 
    print("\n### Step #1 - The Auto MPG dataset: Get the data")

    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']

    raw_dataset = pd.read_csv(
        url, 
        names=column_names,
        na_values='?', 
        comment='\t',
        sep=' ', 
        skipinitialspace=True
    )
    dataset = raw_dataset.copy()

    if args.step == 1:
        logger.info(f'\n{dataset.tail()}')


args.step = auto_increment(args.step, args.all)
### Step #2 - The Auto MPG dataset: Clean the data
if args.step >= 2:
    print("\n### Step #2 - The Auto MPG dataset: Clean the data")

    if args.step == 2:
        logger.info(f'inspect na values:\n{dataset.isna().sum()}\n')

    dataset = dataset.dropna()

    # one-hot encoding
    dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')

    if args.step == 2:
        logger.info(f'after one-hot encoding:\n{dataset.tail()}')

    
args.step = auto_increment(args.step, args.all)
### Step #3 - The Auto MPG dataset: Split the data into train and test
if args.step >= 3:
    print("\n### Step #3 - The Auto MPG dataset: Split the data into train and test")

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    if args.step == 3:
        logger.info(f'train_dataset.shape: {train_dataset.shape}')
        logger.info(f'test_dataset.shape: {test_dataset.shape}')


args.step = auto_increment(args.step, args.all)
### Step #4 - The Auto MPG dataset: Inspect the data
if args.step == 4:
    print("\n### Step #4 - The Auto MPG dataset: Inspect the data")

    if args.plot:
        sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
        plt.show(block=False)

    logger.info('overall statistics for the features:')
    print(train_dataset.describe().transpose())


args.step = auto_increment(args.step, args.all)
### Step #5 - The Auto MPG dataset: Split features from labels
if args.step >= 5:
    print("\n### Step #5 - The Auto MPG dataset: Split features from labels")

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('MPG')
    test_labels = test_features.pop('MPG')


args.step = auto_increment(args.step, args.all)
### Step #6 - Normalization
if args.step >= 6:
    print("\n### Step #6 - Normalization")

    if args.step == 6:
        logger.info('different the ranges of each feature:')
        print(train_dataset.describe().transpose()[['mean', 'std']])
        print()

    # The Normalization layer
    normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
    normalizer.adapt(train_features.to_numpy())

    if args.step == 6:
        logger.info(f'normalizer stores the mean and variation of each feature:\n{normalizer.mean.numpy()}\n')

        first = train_features[:1]
        with np.printoptions(precision=2, suppress=True):
            logger.info(f'Before normalization:\n{first.values}')
            logger.info(f'After Normalized:\n{normalizer(first).numpy()}')


### Plot utilities
if args.step:
    def plot_loss(history):
        plt.figure()
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.ylim([0, 10])
        plt.xlabel('Epoch')
        plt.ylabel('Error [MPG]')
        plt.legend()
        plt.grid(True)
        plt.show(block=False)
        
    def plot_horsepower(x, y):
        plt.figure()
        plt.scatter(train_features['Horsepower'], train_labels, label='Data')
        plt.plot(x, y, color='k', label='Predictions')
        plt.xlabel('Horsepower')
        plt.ylabel('MPG')
        plt.legend()  
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #7 - Linear regression: One variable
if args.step >= 7:
    print("\n### Step #7 - Linear regression: One variable")

    horsepower = np.array(train_features['Horsepower'])
    horsepower_normalizer = Normalization(input_shape=(1,))
    horsepower_normalizer.adapt(horsepower)

    horsepower_model = Sequential([
        horsepower_normalizer,
        Dense(1)
    ])

    if args.step == 7:
        horsepower_model.summary()
        print()

        # run the untrained model
        preds = horsepower_model.predict(horsepower[:10])
        logger.info(f'untrained model predictions:\n{preds.reshape((10,))}\n')

    horsepower_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error'
    )

    start = time.time()
    history = horsepower_model.fit(
        train_features['Horsepower'], 
        train_labels,
        epochs=args.epochs,
        verbose=0, 
        # Calculate validation results on 20% of the training data
        validation_split=0.2
    )
    end = time.time()

    if args.step == 7:
        logger.info(f"Time taken of horsepower_model is {end - start:.2f} secs\n")

        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        logger.info(f'history\n{hist.tail()}')

        if args.plot:
            plot_loss(history)
            x = tf.linspace(0.0, 250, 251)
            y = horsepower_model.predict(x)
            plot_horsepower(x,y)


args.step = auto_increment(args.step, args.all)
### Step #8 - Linear regression: Multiple inputs
if args.step >= 8:
    print("\n### Step #8 - Linear regression: Multiple inputs")

    linear_model = Sequential([
        normalizer,
        Dense(1)
    ])

    linear_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error'
    )

    start = time.time()
    history = linear_model.fit(
        train_features, 
        train_labels,
        epochs=args.epochs,
        verbose=0, # suppress logging
        validation_split=0.2
    )
    end = time.time()

    if args.step == 8: 
        logger.info(f"Time taken of linear_model is {end - start:.2f} secs")
        if args.plot:
            plot_loss(history)


args.step = auto_increment(args.step, args.all)
### Step #9 - A DNN regression
if args.step >= 9:
    print("\n### Step #9 - A DNN regression")
    
    def build_and_compile_model(norm):
        model = Sequential([
            norm,
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1)
        ])

        model.compile(
            loss='mean_absolute_error',
            optimizer=tf.keras.optimizers.Adam(0.001)
        )
        return model


args.step = auto_increment(args.step, args.all)
### Step #10 - A DNN regression: One variable
if args.step >= 10:
    print("\n### Step #10 - A DNN regression: One variable")

    dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)

    start = time.time()
    history = dnn_horsepower_model.fit(
        train_features['Horsepower'], 
        train_labels,
        epochs=args.epochs,
        verbose=0,
        validation_split=0.2
    )
    end = time.time()

    if args.step == 10:
        logger.info(f"Time taken of dnn_horsepower_model is {end - start:.2f} secs\n")

        dnn_horsepower_model.summary()

        if  args.plot:
            plot_loss(history)

            x = tf.linspace(0.0, 250, 251)
            y = dnn_horsepower_model.predict(x)
            plot_horsepower(x,y)


args.step = auto_increment(args.step, args.all)
### Step #11 - A DNN regression: Full model
if args.step >= 11:
    print("\n### Step #11 - A DNN regression: Full model")

    dnn_model = build_and_compile_model(normalizer)

    start = time.time()
    history = dnn_model.fit(
        train_features, 
        train_labels,
        epochs=args.epochs,
        verbose=0,
        validation_split=0.2
    )
    end = time.time()

    if args.step == 11:
        logger.info(f"Time taken of dnn_model is {end - start:.2f} secs\n")

        dnn_model.summary()

        if args.plot:
            plot_loss(history)


args.step = auto_increment(args.step, args.all)
### Step #12 - Performance
if args.step == 12:
    print("\n### Step #12 - Performance")

    test_results = {}

    test_results['horsepower_model'] = horsepower_model.evaluate(
        test_features['Horsepower'],
        test_labels, 
        verbose=0
    )

    test_results['linear_model'] = linear_model.evaluate(
        test_features,
        test_labels, 
        verbose=2
    )

    test_results['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(
        test_features['Horsepower'],
        test_labels, 
        verbose=0
    )

    test_results['dnn_model'] = dnn_model.evaluate(
        test_features, 
        test_labels, 
        verbose=0
    )

    logger.info(f"Test performance:\n{pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T}\n")

    # make prediction
    test_predictions = dnn_model.predict(test_features).flatten()

    if args.plot:
        plt.figure()
        a = plt.axes(aspect='equal')
        plt.scatter(test_labels, test_predictions)
        plt.xlabel('True Values [MPG]')
        plt.ylabel('Predictions [MPG]')
        lims = [0, 50]
        plt.xlim(lims)
        plt.ylim(lims)
        _ = plt.plot(lims, lims)
        plt.show(block=False)

        # error distribution
        error = test_predictions - test_labels
        plt.figure()
        plt.hist(error, bins=25)
        plt.xlabel('Prediction Error [MPG]')
        _ = plt.ylabel('Count')
        plt.show(block=False)

    # save and reload 
    dnn_model.save('tmp/tf2_t0204/dnn_model')
    reloaded = tf.keras.models.load_model('tmp/tf2_t0204/dnn_model')

    test_results['reloaded'] = reloaded.evaluate(
        test_features, 
        test_labels, 
        verbose=0
    )
    logger.info(f"Reloaded against dnn_model:\n{pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T}")


### End of File
print()
if args.plot:
    plt.show()
debug()

