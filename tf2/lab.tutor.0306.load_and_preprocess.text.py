#!/usr/bin/env python

# Be sure you're using the stable versions of both tf and tf-text, for binary compatibility.
# pip install -q -U tf-nightly
# pip install -q -U tensorflow-text-nightly

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

if args.debug:
    import pdb
    import rlcompleter
    pdb.Pdb.complete=rlcompleter.Completer(locals()).complete
    # import code
    # code.interact(local=locals())
    debug = breakpoint

import time
import collections
import pathlib
import re
import string

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("\n#################################################")
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Concatenate, Activation
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import tensorflow_datasets as tfds
import tensorflow_text as tf_text


### TOC
if args.step == 0:
    toc(__file__)


### Common across all steps
if True:
    AUTOTUNE = tf.data.AUTOTUNE
    def configure_dataset(dataset):
        return dataset.cache().prefetch(buffer_size=AUTOTUNE)

    VOCAB_SIZE = 10000
    MAX_SEQUENCE_LENGTH = 250

    BUFFER_SIZE = 50000
    BATCH_SIZE = 64
    VALIDATION_SIZE = 5000

    def create_model(vocab_size, num_labels):
        model = Sequential([
            Embedding(vocab_size, 64, mask_zero=True),
            Conv1D(64, 5, padding="valid", activation="relu", strides=2),
            GlobalMaxPooling1D(),
            Dense(num_labels)
        ])
        return model


args.step = auto_increment(args.step, args.all)
### Step #1 - Predict the tag for a Stack Overflow question: Download and explore the dataset
if args.step in [1, 2, 3, 4, 5, 6, 7]: 
    print("\n### Step #1 - Predict the tag for a Stack Overflow question: Download and explore the dataset")

    data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'
    dataset = tf.keras.utils.get_file(
        'stack_overflow_16k',
        data_url,
        untar=True,
        cache_subdir='datasets/stack_overflow'
    )
    dataset_dir = pathlib.Path(dataset).parent
    train_dir = dataset_dir/'train'

    if args.step == 1:
        logger.info(dataset_dir)
        print(*list(dataset_dir.iterdir()), sep="\n")
        print('')

        logger.info(train_dir)
        print(*list(train_dir.iterdir()), sep="\n")
        print('')

        sample_file = train_dir/'python/1755.txt'
        logger.info(f'{os.path.basename(sample_file)}:')
        with open(sample_file) as f:
            print(f.read())


args.step = auto_increment(args.step, args.all)
### Step #2 - Predict the tag for a Stack Overflow question: Load the dataset
if args.step in [2, 3, 4, 5, 6, 7]: 
    print("\n### Step #2 - Predict the tag for a Stack Overflow question: Load the dataset")

    batch_size = 32
    seed = 42

    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed
    )

    if args.step == 2:
        for text_batch, label_batch in raw_train_ds.take(1):
            for i in range(10):
                print(f"Question: {text_batch.numpy()[i][:30]}... => Label: {label_batch.numpy()[i]}")

        for i, label in enumerate(raw_train_ds.class_names):
            print("Label", i, "corresponds to", label)


    raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed
    )

    test_dir = dataset_dir/'test'
    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        test_dir, batch_size=batch_size
    )


args.step = auto_increment(args.step, args.all)
### Step #3 - Predict the tag for a Stack Overflow question: Prepare the dataset for training
if args.step in [3, 4, 5, 6, 7]: 
    print("\n### Step #3 - Predict the tag for a Stack Overflow question: Prepare the dataset for training")

    binary_vectorize_layer = TextVectorization(
        max_tokens=VOCAB_SIZE, # 10000
        output_mode='binary'
    )

    int_vectorize_layer = TextVectorization(
        max_tokens=VOCAB_SIZE, # 10000
        output_mode='int',
        output_sequence_length=MAX_SEQUENCE_LENGTH # 250
    )

    # Make a text-only dataset (without labels), then call adapt
    train_text = raw_train_ds.map(lambda text, labels: text)
    binary_vectorize_layer.adapt(train_text)
    int_vectorize_layer.adapt(train_text)

    def binary_vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return binary_vectorize_layer(text), label

    def int_vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return int_vectorize_layer(text), label

    if args.step == 3:
        # Retrieve a batch (of 32 reviews and labels) from the dataset
        text_batch, label_batch = next(iter(raw_train_ds))
        first_question, first_label = text_batch[0], label_batch[0]
        logger.info(f"Question: {first_question.numpy()[:30]}")
        logger.info(f"Label: {first_label}")

        logger.info(f"'binary' vectorized question:\n{binary_vectorize_text(first_question, first_label)[0]}")
        logger.info(f"'int' vectorized question:\n{int_vectorize_text(first_question, first_label)[0]}")

        logger.info(f"1289 ---> {int_vectorize_layer.get_vocabulary()[1289]}")
        logger.info(f"313 ---> {int_vectorize_layer.get_vocabulary()[313]}")
        logger.info("Vocabulary size: {}".format(len(int_vectorize_layer.get_vocabulary())))

    binary_train_ds = raw_train_ds.map(binary_vectorize_text)
    binary_val_ds = raw_val_ds.map(binary_vectorize_text)
    binary_test_ds = raw_test_ds.map(binary_vectorize_text)

    int_train_ds = raw_train_ds.map(int_vectorize_text)
    int_val_ds = raw_val_ds.map(int_vectorize_text)
    int_test_ds = raw_test_ds.map(int_vectorize_text)


args.step = auto_increment(args.step, args.all)
### Step #4 - Predict the tag for a Stack Overflow question: Configure the dataset for performance
if args.step in [4, 5, 6, 7]: 
    print("\n### Step #4 - Predict the tag for a Stack Overflow question: Configure the dataset for performance")

    binary_train_ds = configure_dataset(binary_train_ds)
    binary_val_ds = configure_dataset(binary_val_ds)
    binary_test_ds = configure_dataset(binary_test_ds)

    int_train_ds = configure_dataset(int_train_ds)
    int_val_ds = configure_dataset(int_val_ds)
    int_test_ds = configure_dataset(int_test_ds)


args.step = auto_increment(args.step, args.all)
### Step #5 - Predict the tag for a Stack Overflow question: Train the model
if args.step in [5, 6, 7]: 
    print("\n### Step #5 - Predict the tag for a Stack Overflow question: Train the model")

    binary_model = Sequential([Dense(4)])
    binary_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy']
    )
    logger.info('binary_model.fit()')
    history = binary_model.fit(
        binary_train_ds, 
        validation_data=binary_val_ds, 
        epochs=args.epochs,
        verbose=2 if args.step == 5 else 0
    )

    # vocab_size is VOCAB_SIZE + 1 since 0 is used additionally for padding.
    int_model = create_model(vocab_size=VOCAB_SIZE + 1, num_labels=4)
    int_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy']
    )
    logger.info('int_model.fit()')
    history = int_model.fit(
        int_train_ds, 
        validation_data=int_val_ds, 
        epochs=args.epochs,
        verbose=2 if args.step == 5 else 0
    )

    if args.step == 5:
        # Compare the two models
        logger.info("Linear model on binary vectorized data:")
        binary_model.summary()

        logger.info("ConvNet model on int vectorized data:")
        int_model.summary()

        # Evaluate both models on the test data
        binary_loss, binary_accuracy = binary_model.evaluate(binary_test_ds, verbose=0)
        int_loss, int_accuracy = int_model.evaluate(int_test_ds, verbose=0)

        logger.info("Binary model accuracy: {:2.2%}".format(binary_accuracy))
        logger.info("Int model accuracy: {:2.2%}".format(int_accuracy))


args.step = auto_increment(args.step, args.all)
### Step #6 - Predict the tag for a Stack Overflow question: Export the model
if args.step in [6, 7]: 
    print("\n### Step #6 - Predict the tag for a Stack Overflow question: Export the model")

    export_model = Sequential([
        binary_vectorize_layer, 
        binary_model,
        tf.keras.layers.Activation('sigmoid')
    ])

    export_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer='adam',
        metrics=['accuracy']
    )

    # Test it with `raw_test_ds`, which yields raw strings
    loss, accuracy = export_model.evaluate(raw_test_ds, verbose=0)
    if args.step == 6:
        logger.info("Accuracy: {:2.2%}".format(accuracy))


args.step = auto_increment(args.step, args.all)
### Step #7 - Predict the tag for a Stack Overflow question: Run inference on new data
if args.step == 7: 
    print("\n### Step #7 - Predict the tag for a Stack Overflow question: Run inference on new data")

    def get_string_labels(predicted_scores_batch):
        predicted_int_labels = tf.argmax(predicted_scores_batch, axis=1)
        predicted_labels = tf.gather(raw_train_ds.class_names, predicted_int_labels)
        return predicted_labels

    inputs = [
        "how do I extract keys from a dict into a list?",  # python
        "debug public static void main(string[] args) {...}",  # java
    ]

    predicted_scores = export_model.predict(inputs)
    logger.info(f'predicted_scores:\n{predicted_scores}\n')
    predicted_labels = get_string_labels(predicted_scores)
    for input, label in zip(inputs, predicted_labels):
        print("Question: ", input)
        print("Predicted label: ", label.numpy())
        print('')


args.step = auto_increment(args.step, args.all)
### Step #8 - Predict the author of Illiad translations: Download and explore the dataset
if args.step in [8, 9, 10, 11, 12, 13, 14]: 
    print("\n### Step #8 - Predict the author of Illiad translations: Download and explore the dataset")

    DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
    FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']

    for name in FILE_NAMES:
        text_dir = tf.keras.utils.get_file(
            name, origin=DIRECTORY_URL + name, cache_subdir='datasets/illiad')

    parent_dir = pathlib.Path(text_dir).parent
    if args.step == 8:
        logger.info(f'parent_dir: {parent_dir}\n')


args.step = auto_increment(args.step, args.all)
### Step #9 - Predict the author of Illiad translations: Load the dataset
if args.step in [9, 10, 11, 12, 13, 14]: 
    print("\n### Step #9 - Predict the author of Illiad translations: Load the dataset")

    def labeler(example, index):
        return example, tf.cast(index, tf.int64)

    labeled_data_sets = []

    for i, file_name in enumerate(FILE_NAMES):
        lines_dataset = tf.data.TextLineDataset(str(parent_dir/file_name))
        labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
        labeled_data_sets.append(labeled_dataset)

    all_labeled_data = labeled_data_sets[0]
    for labeled_dataset in labeled_data_sets[1:]:
        all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

    all_labeled_data = all_labeled_data.shuffle(
        BUFFER_SIZE, reshuffle_each_iteration=False
    )

    if args.step == 9:
        for text, label in all_labeled_data.take(10):
            print(f"Sentence: {text.numpy()[:30]}... => Label: {label.numpy()}")
        print('')


args.step = auto_increment(args.step, args.all)
### Step #10 - Predict the author of Illiad translations: Prepare the dataset for training
if args.step in [10, 11, 12, 13, 14]: 
    print("\n### Step #10 - Predict the author of Illiad translations: Prepare the dataset for training")

    tokenizer = tf_text.UnicodeScriptTokenizer()

    # 문장을 소문자로 바꾸고, 단어별로 분리 
    def tokenize(text, unused_label):
        lower_case = tf_text.case_fold_utf8(text)
        return tokenizer.tokenize(lower_case)

    tokenized_ds = all_labeled_data.map(tokenize)

    if args.step == 10:
        logger.info('tokenized_ds.take(5):')
        for text_batch in tokenized_ds.take(5):
            print(text_batch.numpy()[:5], "...")
        print('')

    tokenized_ds = configure_dataset(tokenized_ds)
    vocab_dict = collections.defaultdict(lambda: 0)
    for toks in tokenized_ds.as_numpy_iterator():
        for tok in toks:
            vocab_dict[tok] += 1

    vocab = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
    vocab = [token for token, count in vocab]
    vocab = vocab[:VOCAB_SIZE]
    vocab_size = len(vocab)
    if args.step == 10:
        logger.info("Vocab size: {}".format(vocab_size))
        logger.info("First five vocab entries:\n{}\n".format(vocab[:5]))

    keys = vocab
    values = range(2, len(vocab) + 2)  # reserve 0 for padding, 1 for OOV

    init = tf.lookup.KeyValueTensorInitializer(
        keys, values, key_dtype=tf.string, value_dtype=tf.int64
    )

    num_oov_buckets = 1
    vocab_table = tf.lookup.StaticVocabularyTable(init, num_oov_buckets)

    def preprocess_text(text, label):
        standardized = tf_text.case_fold_utf8(text)
        tokenized = tokenizer.tokenize(standardized)
        vectorized = vocab_table.lookup(tokenized)
        return vectorized, label

    if args.step == 10:
        example_text, example_label = next(iter(all_labeled_data))
        logger.info('Sentence to Vectorized sentence:')
        print("Sentence: {}".format(example_text.numpy()))
        vectorized_text, example_label = preprocess_text(example_text, example_label)
        print("Vectorized sentence: ", vectorized_text.numpy())
        print('')

    all_encoded_data = all_labeled_data.map(preprocess_text)


args.step = auto_increment(args.step, args.all)
### Step #11 - Predict the author of Illiad translations: Split the dataset into train and test
if args.step in [11, 12, 13, 14]: 
    print("\n### Step #11 - Predict the author of Illiad translations: Split the dataset into train and test")

    train_data = all_encoded_data.skip(VALIDATION_SIZE).shuffle(BUFFER_SIZE)
    validation_data = all_encoded_data.take(VALIDATION_SIZE)

    train_data = train_data.padded_batch(BATCH_SIZE)
    validation_data = validation_data.padded_batch(BATCH_SIZE)

    if args.step == 11:
        sample_text, sample_labels = next(iter(validation_data))
        print("Text batch shape: ", sample_text.shape)
        print("Label batch shape: ", sample_labels.shape)
        print("First text example: ", sample_text[0])
        print("First label example: ", sample_labels[0])

    vocab_size += 2
    train_data = configure_dataset(train_data)
    validation_data = configure_dataset(validation_data)


args.step = auto_increment(args.step, args.all)
### Step #12 - Predict the author of Illiad translations: Train the model
if args.step in [12, 13, 14]: 
    print("\n### Step #12 - Predict the author of Illiad translations: Train the model")

    model = create_model(vocab_size=vocab_size, num_labels=3)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = model.fit(
        train_data, 
        validation_data=validation_data, epochs=3,
        verbose=2 if args.step == 12 else 0
    )

    loss, accuracy = model.evaluate(validation_data, verbose=0)
    if args.step == 12:
        logger.info("Loss: {:.2f}".format(loss)) 
        logger.info("Accuracy: {:2.2%}".format(accuracy))


args.step = auto_increment(args.step, args.all)
### Step #13 - Predict the author of Illiad translations: Export the model
if args.step in [13, 14]: 
    print("\n### Step #13 - Predict the author of Illiad translations: Export the model")
    
    preprocess_layer = TextVectorization(
        max_tokens=vocab_size,
        standardize=tf_text.case_fold_utf8,
        split=tokenizer.tokenize,
        output_mode='int',
        output_sequence_length=MAX_SEQUENCE_LENGTH
    )
    preprocess_layer.set_vocabulary(vocab)

    export_model = Sequential([
        preprocess_layer,
        model,
        Activation('sigmoid')
    ])

    export_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer='adam',
        metrics=['accuracy']
    )

    # Create a test dataset of raw strings
    test_ds = all_labeled_data.take(VALIDATION_SIZE).batch(BATCH_SIZE)
    test_ds = configure_dataset(test_ds)
    loss, accuracy = export_model.evaluate(test_ds, verbose=0)
    if args.step == 13:
        logger.info("Loss: {:.2f}".format(loss))
        logger.info("Accuracy: {:2.2%}".format(accuracy))


args.step = auto_increment(args.step, args.all)
### Step #14 - Predict the author of Illiad translations: Run inference on new data
if args.step == 14: 
    print("\n### Step #14 - Predict the author of Illiad translations: Run inference on new data")

    inputs = [
        "Join'd to th' Ionians with their flowing robes,",  # Label: 1
        "the allies, and his armour flashed about him so that he seemed to all",  # Label: 2
        "And with loud clangor of his arms he fell.",  # Label: 0
    ]

    predicted_scores = export_model.predict(inputs)
    predicted_labels = tf.argmax(predicted_scores, axis=1)
    for input, label in zip(inputs, predicted_labels):
        print(f"Question: {input}... => Label: {label.numpy()}")


args.step = auto_increment(args.step, args.all)
### Step #15 - Downloading more datasets using TensorFlow Datasets (TFDS)
if args.step >= 15: 
    print("\n### Step #15 - Downloading more datasets using TensorFlow Datasets (TFDS)")

    train_ds = tfds.load(
        'imdb_reviews',
        split='train',
        batch_size=BATCH_SIZE,
        shuffle_files=True,
        as_supervised=True
    )

    val_ds = tfds.load(
        'imdb_reviews',
        split='train',
        batch_size=BATCH_SIZE,
        shuffle_files=True,
        as_supervised=True
    )

    if args.step == 15:
        for review_batch, label_batch in val_ds.take(1):
            for i in range(5):
                print(f"Review: {review_batch.numpy()[i][:30]}... => Label: {label_batch.numpy()[i]}")


args.step = auto_increment(args.step, args.all)
### Step #16 - Downloading more datasets using TensorFlow Datasets (TFDS): Prepare the dataset for training
if args.step >= 16: 
    print("\n### Step #16 - Downloading more datasets using TensorFlow Datasets (TFDS): Prepare the dataset for training")

    vectorize_layer = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=MAX_SEQUENCE_LENGTH
    )

    # Make a text-only dataset (without labels), then call adapt
    train_text = train_ds.map(lambda text, labels: text)
    vectorize_layer.adapt(train_text)

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    train_ds = train_ds.map(vectorize_text)
    val_ds = val_ds.map(vectorize_text)

    # Configure datasets for performance as before
    train_ds = configure_dataset(train_ds)
    val_ds = configure_dataset(val_ds)


args.step = auto_increment(args.step, args.all)
### Step #17 - Downloading more datasets using TensorFlow Datasets (TFDS): Train the model
if args.step >= 17: 
    print("\n### Step #17 - Downloading more datasets using TensorFlow Datasets (TFDS): Train the model")

    model = create_model(vocab_size=VOCAB_SIZE + 1, num_labels=1)
    if args.step == 17:
        model.summary()

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy']
    )

    history = model.fit(
        train_ds, validation_data=val_ds, 
        epochs=3, 
        verbose=2 if args.step == 17 else 0
    )

    loss, accuracy = model.evaluate(val_ds, verbose=0)
    if args.step == 17:
        logger.info("Loss: {:.2f}".format(loss))
        logger.info("Accuracy: {:2.2%}".format(accuracy))


args.step = auto_increment(args.step, args.all)
### Step #18 - Downloading more datasets using TensorFlow Datasets (TFDS): Export the model
if args.step == 18: 
    print("\n### Step #18 - Downloading more datasets using TensorFlow Datasets (TFDS): Export the model")

    export_model = Sequential([
        vectorize_layer, 
        model,
        Activation('sigmoid')
    ])

    export_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer='adam',
        metrics=['accuracy']
    )

    # 0 --> negative review
    # 1 --> positive review
    inputs = [
        "This is a fantastic movie.",
        "This is a bad movie.",
        "This movie was so bad that it was good.",
        "I will never say yes to watching this movie.",
    ]
    predicted_scores = export_model.predict(inputs)
    predicted_labels = [int(round(x[0])) for x in predicted_scores]
    for input, label in zip(inputs, predicted_labels):
        print("Question: ", input)
        print("Predicted label: ", label)
        print('')


### End of File
if args.plot:
    plt.show()
debug()







