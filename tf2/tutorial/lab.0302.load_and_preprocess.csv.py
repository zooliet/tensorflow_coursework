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
### Step #1 - In memory data
if args.step in [1, 2]: 
    print("\n### Step #1 - In memory data")

    url = "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv"
    abalone_train = pd.read_csv(
        url,
        names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
               "Viscera weight", "Shell weight", "Age"]
    )

    abalone_features = abalone_train.copy()
    abalone_labels = abalone_features.pop('Age')
    abalone_features = np.array(abalone_features)

    if args.step == 1:
        logger.info('abalone_train.head():')
        print(abalone_train.head(), '\n')

        abalone_model = Sequential([
            Dense(64),
            Dense(1)
        ])

        abalone_model.compile(
            loss = tf.keras.losses.MeanSquaredError(),
            optimizer = tf.keras.optimizers.Adam()
        )

        abalone_model.fit(abalone_features, abalone_labels, epochs=args.epochs, verbose=2)
    

args.step = auto_increment(args.step, args.all)
### Step #2 - Basic preprocessing
if args.step == 2: 
    print("\n### Step #2 - Basic preprocessing")

    normalize = preprocessing.Normalization()
    normalize.adapt(abalone_features)
    
    norm_abalone_model = Sequential([
        normalize,
        Dense(64),
        Dense(1)
    ])

    norm_abalone_model.compile(
        loss = tf.keras.losses.MeanSquaredError(),
        optimizer = tf.keras.optimizers.Adam()
    )

    norm_abalone_model.fit(abalone_features, abalone_labels, epochs=args.epochs, verbose=2)


args.step = auto_increment(args.step, args.all)
### Step #3 - Mixed data types
if args.step in [3, 4] : 
    print("\n### Step #3 - Mixed data types")

    titanic = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
    if args.step == 3:
        logger.info('titanic.head():')
        print(titanic.head(), '\n')

    titanic_features = titanic.copy()
    titanic_labels = titanic_features.pop('survived')
    titanic_features_dict = { 
        name: np.array(value) for name, value in titanic_features.items()
    }

    inputs = {}
    for name, column in titanic_features.items():
        dtype = column.dtype
        if dtype == object:
            dtype = tf.string
        else:
            dtype = tf.float32

        inputs[name] = Input(shape=(1,), name=name, dtype=dtype)

    if args.step == 3:
        logger.info(f'{len(inputs)} x Input() will be fed into a model')

    # preprocessing layer 
    numeric_inputs = {
        name:input for (name, input) in inputs.items() if input.dtype==tf.float32
    }
    if args.step == 3:
        logger.info(f'{len(numeric_inputs)} x numeric_inputs(None,)')

    # 4 x (None, 1) => (None, 4)
    x = Concatenate()(list(numeric_inputs.values()))
    if args.step == 3:
        logger.info(f'Concatenate()(numeric_inputs.values()): {x.shape}\n')

    norm = preprocessing.Normalization()
    norm.adapt(np.array(titanic[numeric_inputs.keys()]))

    all_numeric_inputs = norm(x)
    preprocessed_inputs = [all_numeric_inputs]

    # for categorical feature 
    for name, input in inputs.items():
        if input.dtype == tf.float32:
            continue

        # string to number
        lookup = preprocessing.StringLookup(vocabulary=np.unique(titanic_features[name]))
        # number to onehot
        one_hot = preprocessing.CategoryEncoding(num_tokens=lookup.vocabulary_size())

        x = lookup(input)
        x = one_hot(x)
        preprocessed_inputs.append(x)

    preprocessed_inputs_cat = Concatenate()(preprocessed_inputs)
    if args.step == 3:
        logger.info(f'Concatenate()(preprocessed_inputs): {preprocessed_inputs_cat.shape}')

    titanic_preprocessing = Model(inputs, preprocessed_inputs_cat)

    if args.step == 3: 
        if args.plot:
            tf.keras.utils.plot_model(titanic_preprocessing, 'tmp/titanic_processing.png', rankdir="LR", dpi=72, show_shapes=True)
            image = tf.io.read_file('tmp/titanic_processing.png')
            image = tf.image.decode_png(image)
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            plt.show(block=False)

        sample_input = { name: values[:1] for name, values in titanic_features_dict.items() }
        sample_output = titanic_preprocessing(sample_input)
        logger.info('titanic_preprocessing(sample_input):')
        print(f'{[v[0] for v in sample_input.values()]} => ')
        print(sample_output.numpy(), '\n')

    # titanic_model
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

    titanic_model = titanic_model(titanic_preprocessing, inputs)
    titanic_model.fit(
        x=titanic_features_dict, 
        y=titanic_labels, 
        epochs=args.epochs,
        verbose=0
    )

    if args.step == 3:
        titanic_model.save('tmp/test')
        reloaded = tf.keras.models.load_model('tmp/test')
        
        test_features_dict = {name:values[:1] for name, values in titanic_features_dict.items()}

        before = titanic_model(test_features_dict)
        after = reloaded(test_features_dict)
        assert (before-after) < 1e-3
        print(f'prediction of original model: {before}')
        print(f'prediction of reloaded model: {after}')


args.step = auto_increment(args.step, args.all)
### Step #4 - Using tf.data: On in memory data
if args.step == 4: 
    print("\n### Step #4 - Using tf.data: On in memory data")

    # def slices(features):
    #   for i in itertools.count():
    #     # For each feature take index `i`
    #     example = { name: values[i] for name, values in features.items() }
    #     yield example
    #
    # for example in slices(titanic_features_dict):
    #     for name, value in example.items():
    #         print(f"{name:19s}: {value}")
    #     break

    features_ds = tf.data.Dataset.from_tensor_slices(titanic_features_dict)

    for example in features_ds:
        for name, value in example.items():
            print(f"{name:19s}: {value}")
        print('')
        break

    titanic_ds = tf.data.Dataset.from_tensor_slices((titanic_features_dict, titanic_labels))
    titanic_batches = titanic_ds.shuffle(len(titanic_labels)).batch(32)

    titanic_model.fit(titanic_batches, epochs=args.epochs, verbose=2)


args.step = auto_increment(args.step, args.all)
### Step #5 - Using tf.data: From a single file
if args.step in [5, 6]: 
    print("\n### Step #5 - Using tf.data: From a single file")

    titanic_file_path = tf.keras.utils.get_file(
        "train.csv", 
        "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
    )

    titanic_csv_ds = tf.data.experimental.make_csv_dataset(
        titanic_file_path,
        batch_size=5, # Artificially small to make examples easier to show.
        label_name='survived',
        num_epochs=1,
        ignore_errors=True,
    )

    logger.info('titanic_csv_ds:')
    for batch, label in titanic_csv_ds.take(1):
        for key, value in batch.items():
            print(f"{key:20s}: {value}")
            print(f"{'label':20s}: {label}")
        print('')
    #
    traffic_volume_csv_gz = tf.keras.utils.get_file(
        'Metro_Interstate_Traffic_Volume.csv.gz',
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz",
    )

    traffic_volume_csv_gz_ds = tf.data.experimental.make_csv_dataset(
        traffic_volume_csv_gz,
        batch_size=256, 
        label_name='traffic_volume',
        num_epochs=1,
        compression_type="GZIP"
    )

    logger.info('traffic_volume_csv_gz_ds:')
    for batch, label in traffic_volume_csv_gz_ds.take(1):
        for key, value in batch.items():
            print(f"{key:20s}: {value[:5]}")
            print(f"{'label':20s}: {label[:5]}")
        print('')


args.step = auto_increment(args.step, args.all)
### Step #6 - Using tf.data: Caching 
if args.step == 6: 
    print("\n### Step #6 - Using tf.data: Caching")

    start = time.time()
    for i, (batch, label) in enumerate(traffic_volume_csv_gz_ds.repeat(20)):
        if i % 40 == 0:
            print('.', end='')
        print()
    end = time.time()
    elapsed= end - start

    start = time.time()
    caching = traffic_volume_csv_gz_ds.cache() #.shuffle(1000)
    for i, (batch, label) in enumerate(caching.shuffle(1000).repeat(20)):
        if i % 40 == 0:
            print('.', end='')
        print()
    end = time.time()
    elapsed_caching = end - start
    # Note: Dataset.cache stores the data form the first epoch and replays it in order. 
    # So using .cache disables any shuffles earlier in the pipeline. 
    # Below the .shuffle is added back in after .cache.

    logger.info(f"Time taken is {elapsed:.2f} secs")
    logger.info(f"Time taken with caching is {elapsed_caching:.2f} secs")


args.step = auto_increment(args.step, args.all)
### Step #7 - Using tf.data: Multiple files
if args.step == 7: 
    print("\n### Step #7 - Using tf.data: Multiple files")

    fonts_zip = tf.keras.utils.get_file(
        'fonts.zip',  
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00417/fonts.zip",
        extract=True, cache_subdir='datasets/fonts'
    )

    font_csvs =  sorted(str(p) for p in pathlib.Path(os.path.dirname(fonts_zip)).glob("*.csv"))
    logger.info('font_csvs[:10]:')
    print(*font_csvs[:10], sep='\n')
    print('')
    logger.info(f'len(font_csvs): {len(font_csvs)}') 

    fonts_ds = tf.data.experimental.make_csv_dataset(
        file_pattern = f"{os.path.dirname(fonts_zip)}/*.csv",
        batch_size=10, 
        num_epochs=1,
        num_parallel_reads=20,
        shuffle_buffer_size=10000
    )

    logger.info('fonts_ds:')
    for features in fonts_ds.take(1):
        for i, (name, value) in enumerate(features.items()):
            print(f"{name:20s}: {value}")
            if i > 15:
                break
    print("...")
    logger.info(f"[total: {len(features)} features]")

    # Optional: Packing fields
    def make_images(features):
        image = [None]*400

        new_feats = {}
        for name, value in features.items():
            match = re.match('r(\d+)c(\d+)', name)
            if match:
                image[int(match.group(1))*20+int(match.group(2))] = value
            else:
                new_feats[name] = value

        image = tf.stack(image, axis=0)
        image = tf.reshape(image, [20, 20, -1])
        new_feats['image'] = image

        return new_feats

    fonts_image_ds = fonts_ds.map(make_images)
    for features in fonts_image_ds.take(1):
        break

    if args.plot:
        plt.figure(figsize=(6,6), dpi=120)
        for n in range(9):
            plt.subplot(3,3,n+1)
            plt.imshow(features['image'][..., n])
            plt.title(chr(features['m_label'][n]))
            plt.axis('off')


args.step = auto_increment(args.step, args.all)
### Step #8 - Lower level functions: tf.io.decode_csv
if args.step == 8: 
    print("\n### Step #8 - Lower level functions: tf.io.decode_csv")


args.step = auto_increment(args.step, args.all)
### Step #9 - Lower level functions: tf.data.experimental.CsvDataset
if args.step == 9: 
    print("\n### Step #9 - Lower level functions: tf.data.experimental.CsvDataset")


### End of File
if args.plot:
    plt.show()
debug()

