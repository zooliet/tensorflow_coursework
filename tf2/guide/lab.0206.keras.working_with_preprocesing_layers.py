#!/usr/bin/env python

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
from PIL import Image

from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import Layer, Dense, LSTM, Embedding
from tensorflow.keras.layers.experimental import preprocessing

### Note
# The Keras preprocessing layers API allows developers to build Keras-native 
# input processing pipelines. These input processing pipelines can be used as 
# independent preprocessing code in non-Keras workflows, combined directly with 
# Keras models, and exported as part of a Keras SavedModel.


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - The adapt() method
if args.step == 1:
    print("\n### Step #1 - The adapt() method")

    data = np.array([[0.1, 0.2, 0.3], [0.8, 0.9, 1.0], [1.5, 1.6, 1.7],])
    layer = preprocessing.Normalization()
    layer.adapt(data)
    normalized_data = layer(data)

    logger.info(f"normalized_data:\n{normalized_data}")
    logger.info("normalized_data.mean(): %.2f" % (normalized_data.numpy().mean()))
    logger.info("normalized_datea.std(): %.2f\n" % (normalized_data.numpy().std()))

    ##
    data = [
        "ξεῖν᾽, ἦ τοι μὲν ὄνειροι ἀμήχανοι ἀκριτόμυθοι",
        "γίγνοντ᾽, οὐδέ τι πάντα τελείεται ἀνθρώποισι.",
        "δοιαὶ γάρ τε πύλαι ἀμενηνῶν εἰσὶν ὀνείρων:",
        "αἱ μὲν γὰρ κεράεσσι τετεύχαται, αἱ δ᾽ ἐλέφαντι:", "τῶν οἳ μέν κ᾽ ἔλθωσι διὰ πριστοῦ ἐλέφαντος,",
        "οἵ ῥ᾽ ἐλεφαίρονται, ἔπε᾽ ἀκράαντα φέροντες:",
        "οἱ δὲ διὰ ξεστῶν κεράων ἔλθωσι θύραζε,",
        "οἵ ῥ᾽ ἔτυμα κραίνουσι, βροτῶν ὅτε κέν τις ἴδηται.",
    ]
    layer = preprocessing.TextVectorization()
    layer.adapt(data)
    vectorized_text = layer(data)
    logger.info(f'vectorized_text:\n{vectorized_text}\n')

    ##
    vocab = ["a", "b", "c", "d"]
    data = tf.constant([["a", "c", "d"], ["d", "z", "b"], ["e", "f", "g"]])
    layer = preprocessing.StringLookup(vocabulary=vocab)
    vectorized_data = layer(data)
    logger.info(f'vectorized_data:\n{vectorized_data}\n')


args.step = auto_increment(args.step, args.all)
### Step #2 - Preprocessing data before the model or inside the model 
if args.step == 2:
    print("\n### Step #2 - Preprocessing data before the model or inside the model")

    # Make them part of the model
    logger.info("Make them part of the model:")
    str = '''
    inputs = Input(shape=input_shape)
    x = preprocessing_layer(inputs)
    outputs = rest_of_the_model(x)
    model = Model(inputs, outputs)
    '''
    print(str)

    # or apply it to your tf.data.Dataset
    logger.info("Or apply it to your tf.data.Dataset:")
    str = '''
    dataset = dataset.map(lambda x, y: (preprocessing_layer(x), y))
    '''
    print(str) 


args.step = auto_increment(args.step, args.all)
### Step #3 - Benefits of doing preprocessing inside the model at inference time
if args.step == 3:
    print("\n### Step #3 - Benefits of doing preprocessing inside the model at inference time")

    # The key benefit to doing this is that it makes your model portable and 
    # it helps reduce the training/serving skew.
    str = '''
    inputs = Input(shape=input_shape)
    x = preprocessing_layer(inputs)
    outputs = training_model(x)
    inference_model = Model(inputs, outputs)
    '''
    print(str)


args.step = auto_increment(args.step, args.all)
### Step #4 - Quick recipes: Image data augmentation (on-device)
if args.step == 4:
    print("\n### Step #4 - Quick recipes: Image data augmentation (on-device)")

    # Create a data augmentation stage with horizontal flipping, rotations, zooms
    data_augmentation = Sequential([
        preprocessing.RandomFlip("horizontal"),
        preprocessing.RandomRotation(0.1),
        preprocessing.RandomZoom(0.1),
    ])

    # Create a model that includes the augmentation stage
    input_shape = (32, 32, 3)
    classes = 10
    inputs = Input(shape=input_shape)
    # Augment images
    x = data_augmentation(inputs)
    # Rescale image values to [0, 1]
    x = preprocessing.Rescaling(1.0 / 255)(x)
    # Add the rest of the model
    outputs = tf.keras.applications.ResNet50(
        weights=None, input_shape=input_shape, classes=classes)(x)
    model = Model(inputs, outputs)


args.step = auto_increment(args.step, args.all)
### Step #5 - Quick recipes: Normalizing numerical features
if args.step == 5:
    print("\n### Step #5 - Quick recipes: Normalizing numerical features")

    # Load some data
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.reshape((len(x_train), -1))
    input_shape = x_train.shape[1:]
    classes = 10

    # Create a Normalization layer and set its internal state using the training data
    normalizer = preprocessing.Normalization()
    normalizer.adapt(x_train)

    # Create a model that include the normalization layer
    inputs = Input(shape=input_shape)
    x = normalizer(inputs)
    outputs = Dense(classes, activation="softmax")(x)
    model = Model(inputs, outputs)

    # Train the model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    model.fit(x_train, y_train, verbose=2)


args.step = auto_increment(args.step, args.all)
### Step #6 - Quick recipes: Encoding string categorical features via one-hot encoding
if args.step == 6:
    print("\n### Step #6 - Quick recipes: Encoding string categorical features via one-hot encoding")

    # Define some toy data
    data = tf.constant(["a", "b", "c", "b", "c", "a"])

    # Use StringLookup to build an index of the feature values
    indexer = preprocessing.StringLookup()
    indexer.adapt(data)

    # Use CategoryEncoding to encode the integer indices to a one-hot vector
    encoder = preprocessing.CategoryEncoding(output_mode="binary", num_tokens=5)
    encoder.adapt(indexer(data))

    # Convert new test data (which includes unknown feature values)
    test_data = tf.constant(["a", "b", "c", "d", "e", ""])
    test_data_index = indexer(test_data) 
    encoded_data = encoder(test_data_index)

    logger.info(f'test_data: {test_data.numpy()}')
    logger.info(f'indexer(test_data): {test_data_index.numpy()}')
    logger.info(f'encoded_data: \n{encoded_data.numpy()}\n')


args.step = auto_increment(args.step, args.all)
### Step #7 - Quick recipes: Encoding integer categorical features via one-hot encoding
if args.step == 7:
    print("\n### Step #7 - Quick recipes: Encoding integer categorical features via one-hot encoding")

    # Define some toy data
    data = tf.constant([10, 20, 20, 10, 30, 0])

    # Use IntegerLookup to build an index of the feature values
    indexer = preprocessing.IntegerLookup()
    indexer.adapt(data)

    # Use CategoryEncoding to encode the integer indices to a one-hot vector
    encoder = preprocessing.CategoryEncoding(output_mode="binary", num_tokens=5)
    encoder.adapt(indexer(data))

    # Convert new test data (which includes unknown feature values)
    test_data = tf.constant([10, 10, 20, 50, 60, 0])
    test_data_index = indexer(test_data) 
    encoded_data = encoder(indexer(test_data))

    logger.info(f'test_data: {test_data.numpy()}')
    logger.info(f'indexer(test_data): {test_data_index.numpy()}')
    logger.info(f'encoded_data: \n{encoded_data.numpy()}\n')


args.step = auto_increment(args.step, args.all)
### Step #8 - Quick recipes: Applying the hashing trick to an integer categorical feature
if args.step == 8:
    print("\n### Step #8 - Quick recipes: Applying the hashing trick to an integer categorical feature")

    # Sample data: 10,000 random integers with values between 0 and 100,000
    data = np.random.randint(0, 100000, size=(10000, 1))

    # Use the Hashing layer to hash the values to the range [0, 64]
    hasher = preprocessing.Hashing(num_bins=64, salt=1337)
    data_hashed = hasher(data)

    # Use the CategoryEncoding layer to one-hot encode the hashed values
    encoder = preprocessing.CategoryEncoding(num_tokens=64, output_mode="binary")
    encoded_data = encoder(data_hashed)
    logger.info(encoded_data.shape)

    logger.info(f'data[0] is {data[0]},')
    logger.info(f'hashed into {data_hashed[0]}, and then encoded as \n{encoded_data[0]}\n')


args.step = auto_increment(args.step, args.all)
### Step #9 - Quick recipes: Encoding text as a sequence of token indices
if args.step == 9:
    print("\n### Step #9 - Quick recipes: Encoding text as a sequence of token indices")

    # Define some text data to adapt the layer
    data = tf.constant(
        [
            "The Brain is wider than the Sky",
            "For put them side by side",
            "The one the other will contain",
            "With ease and You beside",
        ]
    )
    # Instantiate TextVectorization with "int" output_mode
    text_vectorizer = preprocessing.TextVectorization(output_mode="int")
    # Index the vocabulary via `adapt()`
    text_vectorizer.adapt(data)

    # You can retrieve the vocabulary we indexed via get_vocabulary()
    vocab = text_vectorizer.get_vocabulary()
    logger.info("Vocabulary: {}".format(vocab))

    # Create an Embedding + LSTM model
    inputs = Input(shape=(1,), dtype="string")
    x = text_vectorizer(inputs)
    x = Embedding(input_dim=len(vocab), output_dim=64)(x)
    outputs = LSTM(1)(x)
    model = Model(inputs, outputs)

    # Call the model on test data (which includes unknown tokens)
    test_data = tf.constant(["The Brain is deeper than the sea"])
    test_output = model(test_data)
    logger.info("Input: {}".format(test_data))
    logger.info("Model output: {}\n".format(test_output))


args.step = auto_increment(args.step, args.all)
### Step #10 - Quick recipes: Encoding text as a dense matrix of ngrams with multi-hot encoding
if args.step == 10:
    print("\n### Step #10 - Quick recipes: Encoding text as a dense matrix of ngrams with multi-hot encoding")

    # Define some text data to adapt the layer
    data = tf.constant(
        [
            "The Brain is wider than the Sky",
            "For put them side by side",
            "The one the other will contain",
            "With ease and You beside",
        ]
    )

    # Instantiate TextVectorization with "binary" output_mode (multi-hot)
    # and ngrams=2 (index all bigrams)
    text_vectorizer = preprocessing.TextVectorization(output_mode="binary", ngrams=2)
    # Index the bigrams via `adapt()`
    text_vectorizer.adapt(data)

    logger.info(
        "Encoded text:\n{}\n".format(
            text_vectorizer(["The Brain is deeper than the sea"]).numpy())
    )

    # Create a Dense model
    inputs = Input(shape=(1,), dtype="string")
    x = text_vectorizer(inputs)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)

    # Call the model on test data (which includes unknown tokens)
    test_data = tf.constant(["The Brain is deeper than the sea"])
    test_output = model(test_data)
    logger.info("Model output: {}\n".format(test_output))


args.step = auto_increment(args.step, args.all)
### Step #11 - Quick recipes: Encoding text as a dense matrix of ngrams with TF-IDF weightin
if args.step == 11:
    print("\n### Step #11 - Quick recipes: Encoding text as a dense matrix of ngrams with TF-IDF weightin")

    # Define some text data to adapt the layer
    data = tf.constant(
        [
            "The Brain is wider than the Sky",
            "For put them side by side",
            "The one the other will contain",
            "With ease and You beside",
        ]
    )
    # Instantiate TextVectorization with "tf-idf" output_mode
    # (multi-hot with TF-IDF weighting) and ngrams=2 (index all bigrams)
    text_vectorizer = preprocessing.TextVectorization(output_mode="tf-idf", ngrams=2)
    # Index the bigrams and learn the TF-IDF weights via `adapt()`
    text_vectorizer.adapt(data)

    logger.info(
        "Encoded text:\n{}\n".format(
            text_vectorizer(["The Brain is deeper than the sea"]).numpy())
    )

    # Create a Dense model
    inputs = Input(shape=(1,), dtype="string")
    x = text_vectorizer(inputs)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)

    # Call the model on test data (which includes unknown tokens)
    test_data = tf.constant(["The Brain is deeper than the sea"])
    test_output = model(test_data)
    logger.info("Model output: {}\n".format(test_output))


### End of File
if args.plot:
    plt.show()
debug()

