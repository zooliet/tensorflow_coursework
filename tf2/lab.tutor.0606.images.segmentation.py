#!/usr/bin/env python

# pip install -q git+https://github.com/tensorflow/examples.git

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

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("\n#################################################")
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Softmax
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - Download the Oxford-IIIT Pets dataset
if args.step in [1, 2, 3, 4]:
    print("\n### Step #1 - Download the Oxford-IIIT Pets dataset")

    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

    if args.step == 1:
        logger.info('dataset:')
        for key, value in dataset.items():
            print(f'{key}:\n{value.element_spec}\n')
        logger.info(f'info:\n{info}\n')

    def normalize(input_image, input_mask):
        input_image = tf.cast(input_image, tf.float32) / 255.0
        input_mask -= 1 # 1,2,3 => 0,1,2
        return input_image, input_mask

    @tf.function
    def load_image_train(datapoint):
        input_image = tf.image.resize(datapoint['image'], (128, 128))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)

        input_image, input_mask = normalize(input_image, input_mask)

        return input_image, input_mask
    
    def load_image_test(datapoint):
        input_image = tf.image.resize(datapoint['image'], (128, 128))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

        input_image, input_mask = normalize(input_image, input_mask)

        return input_image, input_mask

    TRAIN_LENGTH = info.splits['train'].num_examples # 3680
    BATCH_SIZE = 64
    BUFFER_SIZE = 1000
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE # 57

    train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    test = dataset['test'].map(load_image_test)

    train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test.batch(BATCH_SIZE)

    def display(display_list):
        plt.figure()
        title = ['Input Image', 'True Mask', 'Predicted Mask']
        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
            plt.axis('off')
        plt.show(block=False)

    if args.step == 1 and args.plot:
        for image, mask in train.take(2):
            sample_image, sample_mask = image, mask
            display([sample_image, sample_mask])


args.step = auto_increment(args.step, args.all)
### Step #2 - Define the model
if args.step in [2, 3, 4]:
    print("\n### Step #2 - Define the model")

    OUTPUT_CHANNELS = 3

    # encoder/downsampler is a pretrained MobileNetV2 model
    base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = False

    # The decoder/upsampler is simply a series of upsample blocks 
    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]

    def unet_model(output_channels):
        inputs = tf.keras.layers.Input(shape=[128, 128, 3])

        # Downsampling through the model
        skips = down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            output_channels, 
            3, 
            strides=2,
            padding='same'  # 64x64 -> 128x128
        )

        x = last(x)
        return tf.keras.Model(inputs=inputs, outputs=x)


args.step = auto_increment(args.step, args.all)
### Step #3 - Train the model
if args.step in [3, 4]:
    print("\n### Step #3 - Train the model")

    model = unet_model(OUTPUT_CHANNELS)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    tf.keras.utils.plot_model(model, 'tmp/segmentation.png', show_shapes=True)
    if args.plot:
        image = Image.open('tmp/segmentation.png')
        plt.figure()
        plt.imshow(image)
        plt.show(block=False)

    def create_mask(pred_mask):
        pred_mask = tf.argmax(pred_mask, axis=-1) # (1,128,128,3) => (1,128,128)
        pred_mask = pred_mask[..., tf.newaxis] # (1,128,128) => (1,128,128,1)
        return pred_mask[0] # (128,128,1)

    def show_predictions(dataset=None, num=1):
        if dataset:
            for image, mask in dataset.take(num):
                pred_mask = model.predict(image)
                display([image[0], mask[0], create_mask(pred_mask)])
        else:
            for image, mask in train.take(1):
                sample_image, sample_mask = image, mask
                display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])

    class DisplayCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # clear_output(wait=True)
            show_predictions()
            print('\nSample Prediction after epoch {}\n'.format(epoch+1))

    EPOCHS = args.epochs
    VAL_SUBSPLITS = 5
    VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

    if args.step == 3 and args.plot:
        callbacks = [DisplayCallback()]
    else:
        callbacks = []

    model_history = model.fit(
        train_dataset, 
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        validation_data=test_dataset,
        callbacks=callbacks,
        verbose=2 if args.step == 3 else 0
    )

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    if args.step == 3 and args.plot:
        plt.figure()
        plt.plot(model_history.epoch, loss, 'r', label='Training loss')
        plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.ylim([0, 1])
        plt.legend()
        plt.show(block=False)

args.step = auto_increment(args.step, args.all)
### Step #4 - Make predictions
if args.step == 4:
    print("\n### Step #4 - Make predictions")

    if args.plot:
        show_predictions(test_dataset, 3)


args.step = auto_increment(args.step, args.all)
### Step #5 - Optional: Imbalanced classes and class weights
if args.step == 5:
    print("\n### Step #5 - Optional: Imbalanced classes and class weights")


args.step = auto_increment(args.step, args.all)
### Step #6 - Next steps
if args.step == 6:
    print("\n### Step #6 - Next steps")

    logger.info("https://github.com/tensorflow/models/tree/master/research/object_detection")


### End of File
if args.plot:
    plt.show()
debug()


