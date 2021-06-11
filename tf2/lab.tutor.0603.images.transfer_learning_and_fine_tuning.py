#!/usr/bin/env python

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
from PIL import Image
import pathlib

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("\n#################################################")
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Softmax
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - Data preprocessing: Data download
if args.step >= 1:
    print("\n### Step #1 - Data preprocessing: Data download")

    _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    path_to_zip = tf.keras.utils.get_file(
        'cats_and_dogs.zip', origin=_URL, extract=True
    )
    
    PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
    # PATH = os.path.join(str(pathlib.Path(path_to_zip).parent), 'cats_and_dogs_filtered')

    train_dir = os.path.join(PATH, 'train')
    validation_dir = os.path.join(PATH, 'validation')

    BATCH_SIZE = 32
    IMG_SIZE = (160, 160)

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE
    )

    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        validation_dir,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE
    )

    class_names = train_dataset.class_names
    
    if args.step == 1 and args.plot:
        plt.figure(figsize=(10, 10))
        for images, labels in train_dataset.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")
            plt.show(block=False)

    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5) # 5 == 20%
    validation_dataset = validation_dataset.skip(val_batches // 5)
    
    if args.step == 1:
        logger.info(f'class_names: {class_names}')
        logger.info('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
        logger.info('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))


args.step = auto_increment(args.step, args.all)
### Step #2 - Data preprocessing: Configure the dataset for performance
if args.step >= 2:
    print("\n### Step #2 - Data preprocessing: Configure the dataset for performance")

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


args.step = auto_increment(args.step, args.all)
### Step #3 - Data preprocessing: Use data augmentation
if args.step >= 3:
    print("\n### Step #3 - Data preprocessing: Use data augmentation")

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    if args.step == 3 and args.plot:
        for image, _ in train_dataset.take(1):
            plt.figure(figsize=(10, 10))
            first_image = image[0]
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
                plt.imshow(augmented_image[0] / 255)
                plt.axis('off')
            plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #4 - Data preprocessing: Rescale pixel values
if args.step >= 4:
    print("\n### Step #4 - Data preprocessing: Rescale pixel values")

    # (0, 255) to (-1,1)
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    # or 
    # rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)


args.step = auto_increment(args.step, args.all)
### Step #5 - Create the base model from the pre-trained convnets
if args.step >= 5:
    print("\n### Step #5 - Create the base model from the pre-trained convnets")

    # Create the base model from the pre-trained model MobileNet V2
    IMG_SHAPE = IMG_SIZE + (3,) # (160, 160, 3)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights='imagenet'
    )

    if args.step == 5:
        image_batch, label_batch = next(iter(train_dataset))
        feature_batch = base_model(image_batch)
        logger.info(f'feature_batch.shape: {feature_batch.shape}')


args.step = auto_increment(args.step, args.all)
### Step #6 - Feature extraction: Freeze the convolutional base
if args.step >= 6:
    print("\n### Step #6 - Feature extraction: Freeze the convolutional base")
    # freeze the convolutional base before you compile and train the model
    base_model.trainable = False


args.step = auto_increment(args.step, args.all)
### Step #7 - Feature extraction: Important note about BatchNormalization layers
if args.step == 7:
    print("\n### Step #7 - Feature extraction: Important note about BatchNormalization layers")

    # Let's take a look at the base model architecture
    base_model.summary()


args.step = auto_increment(args.step, args.all)
### Step #8 - Feature extraction: Add a classification head
if args.step >= 8:
    print("\n### Step #8 - Feature extraction: Add a classification head")

    global_average_layer = GlobalAveragePooling2D()
    prediction_layer = Dense(1)

    if args.step == 8:
        image_batch, label_batch = next(iter(train_dataset))
        feature_batch = base_model(image_batch) # (32, 160, 160, 3) => (32, 5, 5, 1280)
        feature_batch_average = global_average_layer(feature_batch) # (32, 1280)
        logger.info(f'feature_batch_average.shape: {feature_batch_average.shape}')
        
        prediction_batch = prediction_layer(feature_batch_average) # (32, 1)
        logger.info(f'prediction_batch.shape: {prediction_batch.shape}')

    inputs = Input(shape=(160, 160, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = Model(inputs, outputs)


args.step = auto_increment(args.step, args.all)
### Step #9 - Feature extraction: Compile the model
if args.step >= 9:
    print("\n### Step #9 - Feature extraction: Compile the model")

    base_learning_rate = 0.0001
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    if args.step == 9:
        model.summary()
        logger.info(f'len(model.trainable_variables): {len(model.trainable_variables)}\n')
        # The 2.5M parameters in MobileNet are frozen, 
        # but there are 1.2K trainable parameters in the Dense layer. 
        # These are divided between two tf.Variable objects, the weights and biases.


args.step = auto_increment(args.step, args.all)
### Step #10 - Feature extraction: Train the model
if args.step >= 10:
    print("\n### Step #10 - Feature extraction: Train the model")

    initial_epochs = 10
    loss0, accuracy0 = model.evaluate(validation_dataset, verbose=0)

    if args.step == 10:
        logger.info("initial loss: {:.2f}".format(loss0))
        logger.info("initial accuracy: {:.2f}".format(accuracy0))

    history = model.fit(
        train_dataset,
        epochs=initial_epochs,
        validation_data=validation_dataset,
        verbose=2 if args.step == 10 else 0
    )


args.step = auto_increment(args.step, args.all)
### Step #11 - Feature extraction: Learning curves
if args.step == 11:
    print("\n### Step #11 - Feature extraction: Learning curves")

    if args.plot:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0,1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show(block=False)

        # If you are wondering why the validation metrics are clearly better 
        # than the training metrics, the main factor is because layers like 
        # tf.keras.layers.BatchNormalization and tf.keras.layers.Dropout affect 
        # accuracy during training. They are turned off when calculating validation loss


args.step = auto_increment(args.step, args.all)
### Step #12 - Fine tuning: Un-freeze the top layers of the model
if args.step >= 12:
    print("\n### Step #12 - Fine tuning: Un-freeze the top layers of the model")

    base_model.trainable = True
    
    if args.step == 12:
        # Let's take a look to see how many layers are in the base model
        logger.info("Number of layers in the base model: {}".format(len(base_model.layers)))

    # Fine-tune from this layer onwards
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False


args.step = auto_increment(args.step, args.all)
### Step #13 - Fine tuning: Compile the model
if args.step >= 13:
    print("\n### Step #13 - Fine tuning: Compile the model")

    # use a lower learning rate at this stage
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
        metrics=['accuracy']
    )

    if args.step == 13:
        model.summary()
        logger.info(f'len(model.trainable_variables): {len(model.trainable_variables)}')


args.step = auto_increment(args.step, args.all)
### Step #14 - Fine tuning: Continue training the model
if args.step >= 14:
    print("\n### Step #14 - Fine tuning: Continue training the model")

    fine_tune_epochs = 10
    total_epochs =  initial_epochs + fine_tune_epochs

    history_fine = model.fit(
        train_dataset,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        validation_data=validation_dataset,
        verbose=2 if args.step == 10 else 0
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    acc += history_fine.history['accuracy']
    val_acc += history_fine.history['val_accuracy']

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']

    if args.step == 14 and args.plot:
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.ylim([0.8, 1])
        plt.plot([initial_epochs-1,initial_epochs-1], plt.ylim(), label='Start Fine Tuning')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.ylim([0, 1.0])
        plt.plot([initial_epochs-1,initial_epochs-1], plt.ylim(), label='Start Fine Tuning')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #15 - Fine tuning: Evaluation and prediction
if args.step == 15:
    print("\n### Step #15 - Fine tuning: Evaluation and prediction")

    loss, accuracy = model.evaluate(test_dataset, verbose=0)
    logger.info('Test accuracy : {}'.format(accuracy))

    # Retrieve a batch of images from the test set
    image_batch, label_batch = test_dataset.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch).flatten()

    # Apply a sigmoid since our model returns logits
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)

    logger.info(f'Predictions:\n{predictions.numpy()}')
    logger.info(f'Labels:\n{label_batch}')

    if args.plot:
        plt.figure(figsize=(10, 10))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image_batch[i].astype("uint8"))
            plt.title(class_names[predictions[i]])
            plt.axis("off")
        plt.show(block=False)


### End of File
if args.plot:
    plt.show()
debug()


