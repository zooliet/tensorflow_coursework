#!/usr/bin/env python

import sys
sys.path.append('./')
sys.path.append('../')

from lab_utils import (
    tf, os, np, plt, logger, ap, BooleanAction,
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
from PIL import Image
import pathlib

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Softmax
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import tensorflow_hub as hub


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - An ImageNet classifier: Download the classifier
if args.step in [1, 2, 3, 4, 5]:
    print("\n### Step #1 - An ImageNet classifier: Download the classifier")

    classifier_model ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"

    IMAGE_SHAPE = (224, 224)
    classifier = Sequential([
        hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
    ])


args.step = auto_increment(args.step, args.all)
### Step #2 - An ImageNet classifier: Run it on a single image
if args.step in [2, 3]:
    print("\n### Step #2 - An ImageNet classifier: Run it on a single image")

    grace_hopper = tf.keras.utils.get_file(
        'image.jpg',
        'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg'
    )

    # read, resize, make matrix, scaling
    grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
    grace_hopper = np.array(grace_hopper)/255.0
    # or
    # grace_hopper = tf.io.read_file(grace_hopper)
    # grace_hopper = tf.image.decode_jpeg(grace_hopper)
    # grace_hopper = tf.image.resize(grace_hopper, IMAGE_SHAPE)
    # grace_hopper = tf.cast(grace_hopper, tf.uint8)
    # or
    # grace_hopper = tf.keras.preprocessing.image.load_img(grace_hopper, target_size=IMAGE_SHAPE)
    # grace_hopper = tf.keras.preprocessing.image.img_to_array(grace_hopper) / 255.0

    result = classifier.predict(grace_hopper[np.newaxis, ...])
    predicted_class = np.argmax(result[0], axis=-1)

    if args.step == 2: 
        logger.info(f'grace_hopper.shape: {grace_hopper.shape}')
        logger.info(f'result.shape: {result.shape}')
        logger.info(f'predicted_class: {predicted_class}')


args.step = auto_increment(args.step, args.all)
### Step #3 - An ImageNet classifier: Decode the predictions
if args.step in [3, 5]:
    print("\n### Step #3 - An ImageNet classifier: Decode the predictions")

    labels_path = tf.keras.utils.get_file(
        'ImageNetLabels.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
    )

    imagenet_labels = np.array(open(labels_path).read().splitlines())

    if args.step == 3 and args.plot:
        plt.imshow(grace_hopper)
        plt.axis('off')
        predicted_class_name = imagenet_labels[predicted_class]
        _ = plt.title("Prediction: " + predicted_class_name.title())
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #4 - Simple transfer learning: Dataset
if args.step >= 4:
    print("\n### Step #4 - Simple transfer learning: Dataset")

    data_root = tf.keras.utils.get_file(
        'flower_photos',
        'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
        untar=True
    )

    batch_size = 32
    img_height = 224
    img_width = 224

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(data_root),
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    class_names = np.array(train_ds.class_names)
    if args.step == 4:
        logger.info(f'class_names: {class_names}')
    
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    for image_batch, labels_batch in train_ds:
        if args.step == 4:
            logger.info(f'image_batch.shape: {image_batch.shape}')
            logger.info(f'labels_batch.shape: {labels_batch.shape}')
        break


args.step = auto_increment(args.step, args.all)
### Step #5 - Simple transfer learning: Run the classifier on a batch of images
if args.step == 5:
    print("\n### Step #5 - Simple transfer learning: Run the classifier on a batch of images")

    result_batch = classifier.predict(train_ds)
    predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]

    if args.plot:
        plt.figure(figsize=(10,9))
        plt.subplots_adjust(hspace=0.5)
        for n in range(30):
            plt.subplot(6,5,n+1)
            plt.imshow(image_batch[n])
            plt.title(predicted_class_names[n])
            plt.axis('off')
            _ = plt.suptitle("ImageNet predictions")
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #6 - Simple transfer learning: Download the headless model
if args.step >= 6:
    print("\n### Step #6 - Simple transfer learning: Download the headless model")

    feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

    # Create the feature extractor. 
    # Use trainable=False to freeze the variables in the feature extractor layer, 
    # so that the training only modifies the new classifier layer.
    feature_extractor_layer = hub.KerasLayer(
        feature_extractor_model, 
        input_shape=(224, 224, 3), 
        trainable=False
    )

    if args.step == 6:
        feature_batch = feature_extractor_layer(image_batch)
        logger.info(f'feature_batch.shape: {feature_batch.shape}') # (32, 1280)


args.step = auto_increment(args.step, args.all)
### Step #7 - Simple transfer learning: Attach a classification head
if args.step >= 7:
    print("\n### Step #7 - Simple transfer learning: Attach a classification head")

    num_classes = len(class_names) # 5
    model = Sequential([
        feature_extractor_layer,
        Dense(num_classes)
    ])

    if args.step == 7:
        model.summary()

        predictions = model(image_batch)
        logger.info(f'predctions.shape: {predictions.shape}')


args.step = auto_increment(args.step, args.all)
### Step #8 - Simple transfer learning: Train the model
if args.step >= 8:
    print("\n### Step #8 - Simple transfer learning: Train the model")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['acc']
    )

    # use a custom callback to log the loss and accuracy of each batch individually, 
    # instead of the epoch average
    class CollectBatchStats(tf.keras.callbacks.Callback):
        def __init__(self):
            self.batch_losses = []
            self.batch_acc = []

        def on_train_batch_end(self, batch, logs=None):
            self.batch_losses.append(logs['loss'])
            self.batch_acc.append(logs['acc'])
            self.model.reset_metrics()

    batch_stats_callback = CollectBatchStats()

    history = model.fit(
        train_ds, 
        epochs=2,
        callbacks=[batch_stats_callback],
        verbose=2 if args.step == 8 else 0
    )

    if args.step == 8 and args.plot:
        plt.figure()
        plt.ylabel("Loss")
        plt.xlabel("Training Steps")
        plt.ylim([0,2])
        plt.plot(batch_stats_callback.batch_losses)
        plt.show(block=False)

        plt.figure()
        plt.ylabel("Accuracy")
        plt.xlabel("Training Steps")
        plt.ylim([0,1])
        plt.plot(batch_stats_callback.batch_acc)
        plt.show(block=False)
            

args.step = auto_increment(args.step, args.all)
### Step #9 - Simple transfer learning: Check the predictions
if args.step == 9:
    print("\n### Step #9 - Simple transfer learning: Check the predictions")

    predicted_batch = model.predict(image_batch)
    predicted_id = np.argmax(predicted_batch, axis=-1)
    predicted_label_batch = class_names[predicted_id]

    if args.plot:
        plt.figure(figsize=(10,9))
        plt.subplots_adjust(hspace=0.5)
        for n in range(30):
            plt.subplot(6,5,n+1)
            plt.imshow(image_batch[n])
            plt.title(predicted_label_batch[n].title())
            plt.axis('off')
            _ = plt.suptitle("Model predictions")
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #10 - Export your model
if args.step == 10:
    print("\n### Step #10 - Export your model")

    t = time.time()
    export_path = "tmp/saved_models/{}".format(int(t))
    model.save(export_path)
    logger.info(f'export_path: {export_path}')

    reloaded = tf.keras.models.load_model(export_path)
    result_batch = model.predict(image_batch)
    reloaded_result_batch = reloaded.predict(image_batch)

    diff = abs(reloaded_result_batch - result_batch).max()
    logger.info(f'diff between models: {diff}')


### End of File
if args.plot:
    plt.show()
debug()


