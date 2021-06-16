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
from PIL import Image

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Layer, Concatenate, Dense, Dropout, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, UpSampling2D 
from tensorflow.keras.layers import Embedding, LSTM


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - Introduction
if args.step in [1,2,3]:
    print("\n### Step #1 - Introduction")

    inputs = Input(shape=(784,)) # TensorShape([None, 784]), tf.float32

    x = Dense(64, activation='relu')(inputs)
    outputs = Dense(10)(x)
    model = Model(inputs=inputs, outputs=outputs, name='mnist_model')

    if args.step == 1:
        logger.info('inputs: {}, {}'.format(inputs.shape, inputs.dtype))

        model.summary()

        tf.keras.utils.plot_model(model, "tmp/my_first_model_with_shape_info.png", show_shapes=True)
        if args.plot:
            image = Image.open('tmp/my_first_model_with_shape_info.png')
            plt.figure()
            plt.imshow(image)
            plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #2 - Training, evaluation, and inference
if args.step in [2,3]:
    print("\n### Step #2 - Training, evaluation, and inference")

    # load mnist data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.RMSprop(),
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train, y_train, 
        batch_size=args.batch, # 64 
        epochs=args.epochs, 
        validation_split=0.2,
        verbose=2 if args.step == 2 else 0 
    )

    test_scores = model.evaluate(x_test, y_test, verbose=0)
    if args.step == 2:
        logger.info("Test loss: {:.2f}".format(test_scores[0]))
        logger.info("Test accuracy: {:.2f}".format(test_scores[1]))


args.step = auto_increment(args.step, args.all)
### Step #3 - Save and serialize
if args.step == 3:
    print("\n### Step #3 - Save and serialize")

    model.save("tmp/my_model")
    del model
    # Recreate the exact same model purely from the file:
    model = tf.keras.models.load_model("tmp/my_model")


args.step = auto_increment(args.step, args.all)
### Step #4 - Use the same graph of layers to define multiple models
if args.step == 4:
    print("\n### Step #4 - Use the same graph of layers to define multiple models")

    encoder_input = Input(shape=(28, 28, 1), name="img")
    x = Conv2D(16, 3, activation="relu")(encoder_input)
    x = Conv2D(32, 3, activation="relu")(x)
    x = MaxPooling2D(3)(x)
    x = Conv2D(32, 3, activation="relu")(x)
    x = Conv2D(16, 3, activation="relu")(x)
    encoder_output = GlobalMaxPooling2D()(x)
    encoder = Model(encoder_input, encoder_output, name="encoder")
    encoder.summary()

    x = Reshape((4, 4, 1))(encoder_output)
    x = Conv2DTranspose(16, 3, activation="relu")(x)
    x = Conv2DTranspose(32, 3, activation="relu")(x)
    x = UpSampling2D(3)(x)
    x = Conv2DTranspose(16, 3, activation="relu")(x)
    decoder_output = Conv2DTranspose(1, 3, activation="relu")(x)
    autoencoder = Model(encoder_input, decoder_output, name="autoencoder")
    autoencoder.summary()    


args.step = auto_increment(args.step, args.all)
### Step #5 - All models are callable, just like layers
if args.step == 5:
    print("\n### Step #5 - All models are callable, just like layers")

    encoder_input = Input(shape=(28, 28, 1), name="original_img")
    x = Conv2D(16, 3, activation="relu")(encoder_input)
    x = Conv2D(32, 3, activation="relu")(x)
    x = MaxPooling2D(3)(x)
    x = Conv2D(32, 3, activation="relu")(x)
    x = Conv2D(16, 3, activation="relu")(x)
    encoder_output = GlobalMaxPooling2D()(x)
    encoder = Model(encoder_input, encoder_output, name="encoder")
    encoder.summary()

    decoder_input = Input(shape=(16,), name="encoded_img")
    x = Reshape((4, 4, 1))(decoder_input)
    x = Conv2DTranspose(16, 3, activation="relu")(x)
    x = Conv2DTranspose(32, 3, activation="relu")(x)
    x = UpSampling2D(3)(x)
    x = Conv2DTranspose(16, 3, activation="relu")(x)
    decoder_output = Conv2DTranspose(1, 3, activation="relu")(x)
    decoder = Model(decoder_input, decoder_output, name="decoder")
    decoder.summary()

    autoencoder_input = Input(shape=(28, 28, 1), name="img")
    encoded_img = encoder(autoencoder_input)
    decoded_img = decoder(encoded_img)
    autoencoder = Model(autoencoder_input, decoded_img, name="autoencoder")
    autoencoder.summary()

    # ensemble model
    def get_model():
        inputs = Input(shape=(128,))
        outputs = Dense(1)(inputs)
        return Model(inputs, outputs)

    model1 = get_model()
    model2 = get_model()
    model3 = get_model()

    inputs = Input(shape=(128,))
    y1 = model1(inputs)
    y2 = model2(inputs)
    y3 = model3(inputs)

    outputs = tf.keras.layers.average([y1, y2, y3])
    ensemble_model = Model(inputs=inputs, outputs=outputs)


args.step = auto_increment(args.step, args.all)
### Step #6 - Manipulate complex graph topologies: with multiple inputs and outputs
if args.step == 6:
    print("\n### Step #6 - Manipulate complex graph topologies: with multiple inputs and outputs")

    num_tags = 12  # Number of unique issue tags
    num_words = 10000  # Size of vocabulary obtained when preprocessing text data
    num_departments = 4  # Number of departments for predictions

    title_input = Input(shape=(None,), name="title")  # Variable-length sequence of ints
    body_input = Input(shape=(None,), name="body")  # Variable-length sequence of ints
    tags_input = Input(shape=(num_tags,), name="tags")  # Binary vectors of size `num_tags`

    # Embed each word in the title into a 64-dimensional vector
    title_features = Embedding(num_words, 64)(title_input)
    # Embed each word in the text into a 64-dimensional vector
    body_features = Embedding(num_words, 64)(body_input)

    # Reduce sequence of embedded words in the title into a single 128-dimensional vector
    title_features = LSTM(128)(title_features)
    # Reduce sequence of embedded words in the body into a single 32-dimensional vector
    body_features = LSTM(32)(body_features)

    # Merge all available features into a single large vector via concatenation
    x = tf.keras.layers.concatenate([title_features, body_features, tags_input])

    # Stick a logistic regression for priority prediction on top of the features
    priority_pred = Dense(1, name="priority")(x)
    # Stick a department classifier on top of the features
    department_pred = Dense(num_departments, name="department")(x)

    # Instantiate an end-to-end model predicting both priority and department
    model = Model(
        inputs=[title_input, body_input, tags_input],
        outputs=[priority_pred, department_pred],
    )

    tf.keras.utils.plot_model(model, "tmp/multi_input_and_output_model.png", show_shapes=True)
    if args.plot:
        img = Image.open('tmp/multi_input_and_output_model.png')
        plt.figure()
        plt.imshow(img)
        plt.show(block=False)

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(1e-3),
        loss={
            "priority": tf.keras.losses.BinaryCrossentropy(from_logits=True),
            "department": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        },
        loss_weights=[1.0, 0.2],
    )
   
    # Dummy input data
    title_data = np.random.randint(num_words, size=(1280, 10))
    body_data = np.random.randint(num_words, size=(1280, 100))
    tags_data = np.random.randint(2, size=(1280, num_tags)).astype("float32")

    # Dummy target data
    priority_targets = np.random.random(size=(1280, 1))
    dept_targets = np.random.randint(2, size=(1280, num_departments))

    history = model.fit(
        {"title": title_data, "body": body_data, "tags": tags_data},
        {"priority": priority_targets, "department": dept_targets},
        epochs=args.epochs,
        batch_size=32,
        verbose=2 
    )


args.step = auto_increment(args.step, args.all)
### Step #7 - Manipulate complex graph topologies: A toy ResNet model
if args.step == 7:
    print("\n### Step #7 - Manipulate complex graph topologies: A toy ResNet model")

    inputs = Input(shape=(32, 32, 3), name="img")
    x = Conv2D(32, 3, activation="relu")(inputs)
    x = Conv2D(64, 3, activation="relu")(x)
    block_1_output = MaxPooling2D(3)(x)

    x = Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
    x = Conv2D(64, 3, activation="relu", padding="same")(x)
    block_2_output = tf.keras.layers.add([x, block_1_output])

    x = Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
    x = Conv2D(64, 3, activation="relu", padding="same")(x)
    block_3_output = tf.keras.layers.add([x, block_2_output])

    x = Conv2D(64, 3, activation="relu")(block_3_output)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10)(x)

    model = Model(inputs, outputs, name="toy_resnet")
    model.summary()

    tf.keras.utils.plot_model(model, "tmp/mini_resnet.png", show_shapes=True)
    if args.plot:
        img = Image.open('tmp/mini_resnet.png')
        plt.figure()
        plt.imshow(img)
        plt.show(block=False)

    #
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["acc"],
    )

    # We restrict the data to the first 1000 samples so as to limit execution time
    # on Colab. Try to train on the entire dataset until convergence!
    model.fit(
        x_train, 
        y_train, 
        # x_train[:1000],
        # y_train[:1000],
        batch_size=64, 
        epochs=args.epochs, 
        validation_split=0.2,
        verbose=2
    )


args.step = auto_increment(args.step, args.all)
### Step #8 - Shared layers 
if args.step == 8:
    print("\n### Step #8 - Shared layers ")

    # Embedding for 1000 unique words mapped to 128-dimensional vectors
    shared_embedding = Embedding(1000, 128)

    # Variable-length sequence of integers
    text_input_a = Input(shape=(None,), dtype="int32")

    # Variable-length sequence of integers
    text_input_b = Input(shape=(None,), dtype="int32")

    # Reuse the same layer to encode both inputs
    encoded_input_a = shared_embedding(text_input_a)
    encoded_input_b = shared_embedding(text_input_b)


args.step = auto_increment(args.step, args.all)
### Step #9 - Extract and reuse nodes in the graph of layers
if args.step == 9:
    print("\n### Step #9 - Extract and reuse nodes in the graph of layers")

    vgg19 = tf.keras.applications.VGG19()
    features_list = [layer.output for layer in vgg19.layers]
    feat_extraction_model = Model(inputs=vgg19.input, outputs=features_list)
    for feat in feat_extraction_model.output:
        logger.info(f"{feat.name}: {feat.shape}")

    img = np.random.random((1, 224, 224, 3)).astype("float32")
    extracted_features = feat_extraction_model(img)


args.step = auto_increment(args.step, args.all)
### Step #10 - Extend the API using custom layers
if args.step == 10:
    print("\n### Step #10 - Extend the API using custom layers")

    class CustomDense(Layer):
        def __init__(self, units=32):
            super(CustomDense, self).__init__()
            self.units = units

        def build(self, input_shape):
            self.w = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer="random_normal",
                trainable=True,
            )
            self.b = self.add_weight(
                shape=(self.units,), initializer="random_normal", trainable=True
            )

        def call(self, inputs):
            return tf.matmul(inputs, self.w) + self.b

    inputs = Input((4,))
    outputs = CustomDense(10)(inputs)
    model = Model(inputs, outputs)


args.step = auto_increment(args.step, args.all)
### Step #11 - When to use the functional API
if args.step == 11:
    print("\n### Step #11 - When to use the functional API")
    logger.info("https://www.tensorflow.org/guide/keras/functional#when_to_use_the_functional_api")


args.step = auto_increment(args.step, args.all)
### Step #12 - Mix-and-match API styles
if args.step == 12:
    print("\n### Step #12 - Mix-and-match API styles")
    pass


### End of File
if args.plot:
    plt.show()
debug()

