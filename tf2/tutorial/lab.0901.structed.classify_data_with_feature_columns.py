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

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Softmax

from sklearn.model_selection import train_test_split


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - Use Pandas to create a dataframe
if args.step >= 1: 
    print("\n### Step #1 - Use Pandas to create a dataframe")

    dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
    csv_file = f'{os.getenv("HOME")}/.keras/datasets/petfinder-mini/petfinder-mini.csv'

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


args.step = auto_increment(args.step, args.all)
### Step #5 - Understand the input pipeline
if args.step == 5: 
    print("\n### Step #5 - Understand the input pipeline")

    features_batch, label_batch = next(iter(train_ds))
    logger.info('sample batch:')
    for name, values in features_batch.items():
        print(f'{name:13s}: {values.numpy()}')
    print('{:13s}: {}'.format('Label', label_batch.numpy()))


args.step = auto_increment(args.step, args.all)
### Step #6 - Demonstrate several types of feature columns
if args.step >= 6: 
    print("\n### Step #6 - Demonstrate several types of feature columns")

    # We will use this batch to demonstrate several types of feature columns
    example_batch = next(iter(train_ds))[0]

    # A utility method to create a feature column
    # and to transform a batch of data
    def demo(feature_column):
        feature_layer = tf.keras.layers.DenseFeatures(feature_column)
        print(feature_layer(example_batch).numpy())


args.step = auto_increment(args.step, args.all)
### Step #7 - Demonstrate several types of feature columns: Numeric columns
if args.step == 7: 
    print("\n### Step #7 - Demonstrate several types of feature columns: Numeric columns")

    logger.info("example_batch['PhotoAmt']:")
    print(example_batch['PhotoAmt'].numpy(), '\n=>')
    photo_count = tf.feature_column.numeric_column('PhotoAmt')
    demo(photo_count)


args.step = auto_increment(args.step, args.all)
### Step #8 - Demonstrate several types of feature columns: Bucketized columns
if args.step in [8, 12]: 
    print("\n### Step #8 - Demonstrate several types of feature columns: Bucketized columns")

    if args.step == 8:
        logger.info("example_batch['Age']:")
        print(example_batch['Age'].numpy(), '\n=>')

    age = tf.feature_column.numeric_column('Age')
    age_buckets = tf.feature_column.bucketized_column(age, boundaries=[1, 3, 5])

    if args.step == 8:
        demo(age_buckets)


args.step = auto_increment(args.step, args.all)
### Step #9 - Demonstrate several types of feature columns: Categorical columns
if args.step in [9, 12]: 
    print("\n### Step #9 - Demonstrate several types of feature columns: Categorical columns")

    if args.step == 9:
        logger.info("example_batch['Type']:")
        print(example_batch['Type'].numpy(), '\n=>')

    animal_type = tf.feature_column.categorical_column_with_vocabulary_list(
        'Type', ['Cat', 'Dog']
    )
    animal_type_one_hot = tf.feature_column.indicator_column(animal_type)

    if args.step == 9:
        demo(animal_type_one_hot)


args.step = auto_increment(args.step, args.all)
### Step #10 - Demonstrate several types of feature columns: Embedding columns
if args.step == 10: 
    print("\n### Step #10 - Demonstrate several types of feature columns: Embedding columns")

    logger.info("example_batch['Breed1']:")
    print(example_batch['Breed1'].numpy(), '\n=>')

    # Notice the input to the embedding column is the categorical column
    # we previously created
    breed1 = tf.feature_column.categorical_column_with_vocabulary_list(
          'Breed1', dataframe.Breed1.unique()
    )
    breed1_embedding = tf.feature_column.embedding_column(breed1, dimension=8)
    demo(breed1_embedding)


args.step = auto_increment(args.step, args.all)
### Step #11 - Demonstrate several types of feature columns: Hashed feature columns
if args.step == 11: 
    print("\n### Step #11 - Demonstrate several types of feature columns: Hashed feature columns")
    
    logger.info("example_batch['Breed1']:")
    print(example_batch['Breed1'].numpy(), '\n=>')

    breed1_hashed = tf.feature_column.categorical_column_with_hash_bucket(
        'Breed1', hash_bucket_size=10
    )
    demo(tf.feature_column.indicator_column(breed1_hashed))


args.step = auto_increment(args.step, args.all)
### Step #12 - Demonstrate several types of feature columns: Crossed feature columns
if args.step == 12: 
    print("\n### Step #12 - Demonstrate several types of feature columns: Crossed feature columns")

    logger.info("example_batch['Age'] and ['Type']:")
    print(example_batch['Age'].numpy())
    print(example_batch['Type'].numpy(), '\n=>')

    crossed_feature = tf.feature_column.crossed_column(
        [age_buckets, animal_type], 
        hash_bucket_size=10
    )
    demo(tf.feature_column.indicator_column(crossed_feature)) 


args.step = auto_increment(args.step, args.all)
### Step #13 - Choose which columns to use
if args.step >= 13: 
    print("\n### Step #13 - Choose which columns to use")

    feature_columns = []

    # numeric cols
    for header in ['PhotoAmt', 'Fee', 'Age']:
        feature_columns.append(tf.feature_column.numeric_column(header))

    # bucketized cols
    age = tf.feature_column.numeric_column('Age')
    age_buckets = tf.feature_column.bucketized_column(age, boundaries=[1, 2, 3, 4, 5])
    feature_columns.append(age_buckets)

    # indicator_columns
    indicator_column_names = [
        'Type', 'Color1', 'Color2', 'Gender', 'MaturitySize',
        'FurLength', 'Vaccinated', 'Sterilized', 'Health'
    ]
    for col_name in indicator_column_names:
        categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
            col_name, dataframe[col_name].unique()
        )
    indicator_column = tf.feature_column.indicator_column(categorical_column)
    feature_columns.append(indicator_column)

    # embedding columns
    breed1 = tf.feature_column.categorical_column_with_vocabulary_list(
        'Breed1', dataframe.Breed1.unique()
    )
    breed1_embedding = tf.feature_column.embedding_column(breed1, dimension=8)
    feature_columns.append(breed1_embedding)

    # crossed columns
    animal_type = tf.feature_column.categorical_column_with_vocabulary_list(
        'Type', ['Cat', 'Dog']
    )
    age_type_feature = tf.feature_column.crossed_column(
        [age_buckets, animal_type], hash_bucket_size=100
    )
    feature_columns.append(tf.feature_column.indicator_column(age_type_feature))

    if args.step == 13:
        logger.info('feature_columns:')
        print(*feature_columns, sep='\n\n')


args.step = auto_increment(args.step, args.all)
### Step #14 - Choose which columns to use: Create a feature layer
if args.step >= 14: 
    print("\n### Step #14 - Choose which columns to use: Create a feature layer")

    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    batch_size = 32
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


args.step = auto_increment(args.step, args.all)
### Step #15 - Create, compile, and train the model
if args.step >= 15: 
    print("\n### Step #15 - Create, compile, and train the model")

    model = Sequential([
        feature_layer,
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dropout(.1),
        Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs, # 10
        verbose=2
    )
    print()

    loss, accuracy = model.evaluate(test_ds, verbose=0)
    logger.info("Loss {:.4f}".format(loss))
    logger.info("Accuracy {:.4f}".format(accuracy))


### End of File
print()
if args.plot:
    plt.show()
debug()


