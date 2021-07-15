#!/usr/bin/env python

# pip install sklearn

import sys
sys.path.append('./')
sys.path.append('../../')

from lab_utils import (
    tf, os, np, plt, logger, ap, BooleanAction,
    debug, toc, auto_increment
)

ap.add_argument('--epochs', type=int, default=10, help='number of epochs: 10*')
# ap.add_argument('--batch', type=int, default=64, help='batch size: 64*')
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
from PIL import Image

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Softmax
from tensorflow.keras.layers.experimental import preprocessing

from sklearn.model_selection import train_test_split


### TOC
if args.step == 0:
    toc(__file__)


if args.step or args.all:
    if not os.path.exists('tmp/tf2_t0902/'):
        os.mkdir('tmp/tf2_t0902/') 


args.step = auto_increment(args.step, args.all)
### Step #1 - Use Pandas to create a dataframe
if args.step >= 1: 
    print("\n### Step #1 - Use Pandas to create a dataframe")

    dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
    csv_file = f'{os.getenv("HOME")}/.keras/datasets/petfinder-mini/petfinder-mini.csv'

    if not os.path.exists(csv_file):
        tf.keras.utils.get_file(
            'petfinder_mini.zip', 
            dataset_url,
            extract=True
        )

    dataframe = pd.read_csv(csv_file)

    if args.step == 1:
        print(dataframe.head())


args.step = auto_increment(args.step, args.all)
### Step #2 - Create target variable
if args.step >= 2: 
    print("\n### Step #2 - Create target variable")

    # In the original dataset "4" indicates the pet was not adopted.
    dataframe['target'] = np.where(dataframe['AdoptionSpeed']==4, 0, 1)

    # Drop un-used columns.
    dataframe = dataframe.drop(columns=['AdoptionSpeed', 'Description'])

    if args.step == 2:
        print(dataframe.head())


args.step = auto_increment(args.step, args.all)
### Step #3 - Split the dataframe into train, validation, and test
if args.step >= 3: 
    print("\n### Step #3 - Split the dataframe into train, validation, and test")

    train, test = train_test_split(dataframe, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)

    if args.step == 3:
        logger.info(f'{len(train)} train examples')
        logger.info(f'{len(val)} validation examples')
        logger.info(f'{len(test)} test examples')


args.step = auto_increment(args.step, args.all)
### Step #4 - Create an input pipeline using tf.data
if args.step >= 4: 
    print("\n### Step #4 - Create an input pipeline using tf.data")

    # A utility method to create a tf.data dataset from a Pandas Dataframe
    def df_to_dataset(dataframe, shuffle=True, batch_size=32):
        dataframe = dataframe.copy()
        labels = dataframe.pop('target')
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        return ds

    batch_size = 5 # A small batch sized is used for demonstration purposes
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    [(train_features, label_batch)] = train_ds.take(1)
    if args.step == 4:
        logger.info('Features:')
        print(*train_features.keys(), sep=', ')
        print()
        print('A batch of ages:', train_features['Age'])
        print('A batch of targets:', label_batch )


args.step = auto_increment(args.step, args.all)
### Step #5 - Demonstrate the use of preprocessing layers: Numeric columns
if args.step >= 5: 
    print("\n### Step #5 - Demonstrate the use of preprocessing layers: Numeric columns")

    def get_normalization_layer(name, dataset):
        # Create a Normalization layer for our feature.
        normalizer = preprocessing.Normalization(axis=None)

        # Prepare a Dataset that only yields our feature.
        feature_ds = dataset.map(lambda x, y: x[name])

        # Learn the statistics of the data.
        normalizer.adapt(feature_ds)

        return normalizer

    if args.step == 5:
        photo_count_col = train_features['PhotoAmt']
        logger.info("train_features['PhotoAmt']:")
        print(photo_count_col.numpy(), "\n=>")
        layer = get_normalization_layer('PhotoAmt', train_ds)
        print(layer(photo_count_col))


args.step = auto_increment(args.step, args.all)
### Step #6 - Demonstrate the use of preprocessing layers: Categorical columns
if args.step >= 6: 
    print("\n### Step #6 - Demonstrate the use of preprocessing layers: Categorical columns")

    def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
        # Create a StringLookup layer which will turn strings into integer indices
        if dtype == 'string':
            index = preprocessing.StringLookup(max_tokens=max_tokens)
        else:
            index = preprocessing.IntegerLookup(max_tokens=max_tokens)

        # Prepare a Dataset that only yields our feature
        feature_ds = dataset.map(lambda x, y: x[name])

        # Learn the set of possible values and assign them a fixed integer index.
        index.adapt(feature_ds)

        # Create a Discretization for our integer indices.
        encoder = preprocessing.CategoryEncoding(num_tokens=index.vocabulary_size())

        # Apply one-hot encoding to our indices. The lambda function captures the
        # layer so we can use them, or include them in the functional model later.
        return lambda feature: encoder(index(feature))

    if args.step == 6:
        type_col = train_features['Type']
        logger.info("train_features['Type']:")
        print(type_col.numpy(), "\n=>")

        layer = get_category_encoding_layer('Type', train_ds, 'string')
        print(layer(type_col))


args.step = auto_increment(args.step, args.all)
### Step #7 - Choose which columns to use
if args.step >= 7: 
    print("\n### Step #7 - Choose which columns to use")

    batch_size = 256
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    all_inputs = []
    encoded_features = []

    # Numeric features.
    for header in ['PhotoAmt', 'Fee']:
        numeric_col = tf.keras.Input(shape=(1,), name=header)
        normalization_layer = get_normalization_layer(header, train_ds)
        encoded_numeric_col = normalization_layer(numeric_col)
        all_inputs.append(numeric_col)
        encoded_features.append(encoded_numeric_col)

    # Categorical features encoded as integers.
    age_col = tf.keras.Input(shape=(1,), name='Age', dtype='int64')
    encoding_layer = get_category_encoding_layer(
        'Age', train_ds, dtype='int64', max_tokens=5
    )
    encoded_age_col = encoding_layer(age_col)
    all_inputs.append(age_col)
    encoded_features.append(encoded_age_col)

    # Categorical features encoded as string.
    categorical_cols = [
        'Type', 'Color1', 'Color2', 'Gender', 'MaturitySize',
        'FurLength', 'Vaccinated', 'Sterilized', 'Health', 'Breed1'
    ]
    for header in categorical_cols:
        categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
        encoding_layer = get_category_encoding_layer(
            header, train_ds, dtype='string', max_tokens=5
        )
        encoded_categorical_col = encoding_layer(categorical_col)
        all_inputs.append(categorical_col)
        encoded_features.append(encoded_categorical_col)


args.step = auto_increment(args.step, args.all)
### Step #8 - Create, compile, and train the model
if args.step >= 8: 
    print("\n### Step #8 - Create, compile, and train the model")

    all_features = tf.keras.layers.concatenate(encoded_features)
    # = all_features = tf.keras.layers.Concatenate()(encoded_features)
    x = Dense(32, activation="relu")(all_features)
    x = Dropout(0.5)(x)
    output = Dense(1)(x)
    model = Model(all_inputs, output)

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    if args.step == 8 and args.plot:
        # rankdir='LR' is used to make the graph horizontal.
        tf.keras.utils.plot_model(
            model, 
            'tmp/tf2_t0902/model.png',
            show_shapes=True, rankdir="LR"
        )

        image = Image.open('tmp/tf2_t0902/model.png')
        plt.figure()
        plt.imshow(image)
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #9 - Create, compile, and train the model: Train the model
if args.step >= 9: 
    print("\n### Step #9 - Create, compile, and train the model: Train the model")

    model.fit(
        train_ds, 
        epochs=args.epochs, # 10, 
        validation_data=val_ds,
        verbose=2 if args.step == 9 else 0
    )

    if args.step == 9:
        print()
        loss, accuracy = model.evaluate(test_ds, verbose=0)
        logger.info("Loss {:.4f}".format(loss))
        logger.info("Accuracy {:.4f}".format(accuracy))


args.step = auto_increment(args.step, args.all)
### Step #10 - Inference on new data
if args.step == 10: 
    print("\n### Step #10 - Inference on new data")

    model.save('tmp/tf2_t0902/my_pet_classifier')
    reloaded_model = tf.keras.models.load_model('tmp/tf2_t0902/my_pet_classifier')

    sample = {
        'Type': 'Cat',
        'Age': 3,
        'Breed1': 'Tabby',
        'Gender': 'Male',
        'Color1': 'Black',
        'Color2': 'White',
        'MaturitySize': 'Small',
        'FurLength': 'Short',
        'Vaccinated': 'No',
        'Sterilized': 'No',
        'Health': 'Healthy',
        'Fee': 100,
        'PhotoAmt': 2,
    }

    input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
    predictions = reloaded_model.predict(input_dict)
    prob = tf.nn.sigmoid(predictions[0])

    print(
        "\nThis particular pet had a %.1f percent probability "
        "of getting adopted." % (100 * prob)
    )


### End of File
print()
if args.plot:
    plt.show()
debug()


