#!/usr/bin/env python

import sys
sys.path.append('./')
sys.path.append('../')

from lab_utils import (
    os, np, plt, logger, ap, BooleanAction,
    debug, toc
)

ap.add_argument('--epochs', type=int, default=10, help='number of epochs: 10*')
ap.add_argument('--batch', type=int, default=32, help='batch size: 32*')
args, extra_args = ap.parse_known_args()
logger.info(args)
# logger.info(extra_args)

if args.debug:
    import pdb
    import rlcompleter
    pdb.Pdb.complete=rlcompleter.Completer(locals()).complete
    # import code
    # code.interact(local=locals())
    debug = breakpoint

import time
import re
import shutil
import string

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("\n#################################################")
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Softmax
from tensorflow.keras.layers import Embedding, Activation, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


### Step #0 - TOC
if args.step == 0:
    toc(__file__)


### Step #1 - Sentiment analysis: Download the IMDB dataset 
if args.step >= 1: 
    print("\n### Step #1 - Sentiment analysis: Download the IMDB dataset ")

    filepath = f'{os.getenv("HOME")}/.keras/datasets/aclImdb_v1.tar.gz'
    if not os.path.exists(filepath):
        filepath = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url, untar=True)
    dataset_dir = os.path.join(os.path.dirname(filepath), 'aclImdb')
    # or
    # url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    # dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True)
    # dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')

    remove_dir = os.path.join(train_dir, 'unsup')
    if os.path.exists(remove_dir):
        shutil.rmtree(remove_dir)

    if args.step == 1:
        logger.info(f'dataset_dir: {dataset_dir}')
        logger.info('ls dataset_dir:')
        print(*os.listdir(dataset_dir), sep=", ")

        logger.info(f'train_dir: {train_dir}')
        logger.info(f'test_dir: {test_dir}')

        logger.info("ls train_dir")
        print(*os.listdir(train_dir), sep=", ")

        sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
        logger.info('cat {}:'.format(sample_file))
        with open(sample_file) as f:
            print(f.read())


### Step #2 - Sentiment analysis: Load the dataset
if args.step >= 2:
    print("\n### Step #2 - Sentiment analysis: Load the dataset")
    
    batch_size=args.batch # 32
    seed = 42

    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed
    )

    raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size, 
        validation_split=0.2, 
        subset='validation', 
        seed=seed
    )

    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        test_dir,
        batch_size=batch_size
    )

    # explore the sample batch
    if args.step == 2:
        for text_batch, label_batch in raw_train_ds.take(1):
            for i in range(1):
                review = text_batch.numpy()[i].decode()
                logger.info("Review: \n{}".format(review))
                logger.info("Label: {}".format(label_batch.numpy()[i]))

    logger.info("Label 0 corresponds to {}".format(raw_train_ds.class_names[0]))
    logger.info("Label 1 corresponds to {}".format(raw_train_ds.class_names[1]))


### Step #3 - Sentiment analysis: Prepare the dataset for training
if args.step >= 3:
    print("\n### Step #3 - Sentiment analysis: Prepare the dataset for training")

    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')

    max_features = 10000
    sequence_length = 250

    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length
    )

    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1) # scalar to vector
        return vectorize_layer(text), label

    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)
    
    # sample
    if args.step == 3:
        text_batch, label_batch = next(iter(raw_train_ds))
        first_review, first_label = text_batch[0], label_batch[0]

        logger.info(f"Review: \n{first_review.numpy().decode()}")
        logger.info(f"Label: {raw_train_ds.class_names[first_label]}")
        logger.info(f"Vectorized review: \n{vectorize_text(first_review, first_label)}")

        logger.info(f"1287 ---> {vectorize_layer.get_vocabulary()[1287]}")
        logger.info(f" 313 ---> {vectorize_layer.get_vocabulary()[313]}")
        logger.info('vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))


### Step #4 - Sentiment analysis: Configure the dataset for performance
if args.step >= 4:
    print("\n### Step #4 - Sentiment analysis: Configure the dataset for performance")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


### Step #5 - Sentiment analysis: Create the model
if args.step >= 5:
    print("\n### Step #5 - Sentiment analysis: Create the model")

    embedding_dim = 16

    model = Sequential([
        Embedding(max_features + 1, embedding_dim),
        Dropout(0.2),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])

    if args.step == 5:
        model.summary()


### Step #6 - Sentiment analysis: Loss function and optimizer
if args.step >= 6:
    print("\n### Step #6 - Sentiment analysis: Loss function and optimizer")

    # Loss function and optimizer
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=tf.keras.metrics.BinaryAccuracy(threshold=0.0)
    )


### Step #7 - Sentiment analysis: Train the model
if args.step >= 7:
    print("\n### Step #7 - Sentiment analysis: Train the model")

    epochs = args.epochs
    # verbose = 2 if args.step == 7 else 0

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2
    )


### Step #8 - Sentiment analysis: Evaluate the model
if args.step == 8:
    print("\n### Step #8 - Sentiment analysis: Evaluate the model")

    loss, accuracy = model.evaluate(test_ds, verbose=0)
    logger.info(f'Loss: {loss:.2f}')
    logger.info(f'Accuracy: {accuracy:.2f}')


### Step #9 - Sentiment analysis: Create a plot of accuracy and loss over time
if args.step == 9:
    print("\n### Step #9 - Sentiment analysis: Create a plot of accuracy and loss over time")

    history_dict = history.history
    logger.info(f'history keys:\n{list(history_dict.keys())}')

    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    if args.plot:
        plt.figure()
        # "bo" is for "blue dot"
        plt.plot(epochs, loss, 'bo', label='Training loss')
        # b is for "solid blue line"
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show(block=False)

        plt.figure()
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show(block=False)


### Step #10 - Export the model
if args.step >= 10:
    print("\n### Step #10 - Export the model")

    export_model = Sequential([
        vectorize_layer,
        model,
        Activation('sigmoid')
    ])

    # train 할 것이 아닌데 compile 할 필요가 있나? => train 상관없이 필요함
    export_model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer="adam",
        metrics=['accuracy']
    )

    # Test it with `raw_test_ds`, which yields raw strings
    loss, accuracy = export_model.evaluate(raw_test_ds, verbose=0)
    logger.info(f'loss: {loss:.4f}')
    logger.info(f'accuracy: {accuracy:.4f}')


### Step #11 - Export the model: Inference on new data
if args.step == 11:
    print("\n### Step #11 - Export the model: Inference on new data")

    # Inference on new data
    examples = [
        "The movie was great!",
        "The movie was okay.",
        "The movie was terrible..."
    ]

    predictions = export_model.predict(examples)
    logger.info(f'Predictions: {predictions.flatten()}')


### End of File
if args.plot:
    plt.show()
debug()

