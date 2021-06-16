#!/usr/bin/env python

# pip install -q pyyaml h5py

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
import shutil

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - Setup: Get an example dataset
if args.step >= 1: 
    print("\n### Step #1 - Setup: Get an example dataset")

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_labels = train_labels[:1000]
    test_labels = test_labels[:1000]

    # (1000, 28, 28) => (1000, 784)
    train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
    test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


args.step = auto_increment(args.step, args.all)
### Step #2 - Setup: Define a model
if args.step >= 2: 
    print("\n### Step #1 - Setup: Define a model")

    def create_model():
        model = Sequential([
            Dense(512, activation='relu', input_shape=(784,)),
            Dropout(0.2),
            Dense(10)
        ])

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )
        return model


    if args.step == 2:
        # Create a basic model instance
        model = create_model()
        model.summary()


args.step = auto_increment(args.step, args.all)
### Step #3 - Save checkpoints during training: Checkpoint callback usage
if args.step == 3:
    print("\n### Step #3 - Save checkpoints during training: Checkpoint callback usage")

    shutil.rmtree('tmp/training_1', ignore_errors=True)
    checkpoint_path = "tmp/training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    logger.info(f'checkpoint_dir: {checkpoint_dir}')

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )

    # Create a basic model instance
    model = create_model()

    # Train the model with the new callback
    logger.info('model.fit()')
    model.fit(
        train_images,
        train_labels,
        epochs=args.epochs,
        validation_data=(test_images, test_labels),
        callbacks=[cp_callback],  # Pass callback to training
        verbose=0
    )
    logger.info('ls -l checkpoint_dir:')
    print(*os.listdir(checkpoint_dir), sep="\n")
    print('')

    # This may generate warnings related to saving the state of the optimizer.
    # These warnings (and similar warnings throughout this notebook)
    # are in place to discourage outdated usage, and can be ignored.

    # new model
    model = create_model()

    # evaluate the untrained model
    loss, acc = model.evaluate(test_images, test_labels, verbose=0)
    logger.info("Untrained model's accuracy: {:5.2f}%".format(100 * acc))

    # Loads the weights
    model.load_weights(checkpoint_path)

    # Re-evaluate the model
    loss, acc = model.evaluate(test_images, test_labels, verbose=0)
    logger.info("Restored model's accuracy: {:5.2f}%".format(100 * acc))


args.step = auto_increment(args.step, args.all)
### Step #4 - Save checkpoints during training: Checkpoint callback options
if args.step == 4:
    print("\n### Step #4 - Save checkpoints during training: Checkpoint callback options")

    # Include the epoch in the file name (uses `str.format`)
    shutil.rmtree('tmp/training_2', ignore_errors=True)
    checkpoint_path = "tmp/training_2/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    logger.info(f'checkpoint_dir: {checkpoint_dir}')

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        # batch_size가 32이고, len(train_images)이 1000 이므로 32x32이 1 epoch에 해당
        # 매 5 epoch 마다 callback을 실행하기 위해서 5x32가 필요
        save_freq=5*32, 
        save_weights_only=True,
        verbose=1
    )

    model = create_model()

    # Save the weights using the `checkpoint_path` format
    model.save_weights(checkpoint_path.format(epoch=0))

    # Train the model with the new callback
    model.fit(
        train_images,
        train_labels,
        epochs=50,
        batch_size=32,
        callbacks=[cp_callback],
        validation_data=(test_images, test_labels),
        verbose=0
    )

    logger.info('ls -l checkpoint_dir:')
    print(*os.listdir(checkpoint_dir), sep="\n")

    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model = create_model()
    model.load_weights(latest)

    # Re-evaluate the model
    loss, acc = model.evaluate(test_images, test_labels, verbose=0)
    logger.info("Restored model's accuracy: {:5.2f}%".format(100 * acc))


args.step = auto_increment(args.step, args.all)
### Step #5 - What are these files?
if args.step == 5:
    print("\n### Step #5 - What are these files?")

    doc = """
    The above code stores the weights to a collection of checkpoint-formatted files
    that contain only the trained weights in a binary format. Checkpoints contain:
    - One or more shards that contain your model's weights.
    - An index file that indicates which weights are stored in which shard.
    If you are training a model on a single machine, 
    you'll have one shard with the suffix: .data-00000-of-00001
    """
    tf.print(doc)


args.step = auto_increment(args.step, args.all)
### Step #6 - Manually save weights
if args.step == 6:
    print("\n### Step #6 - Manually save weights")

    model = create_model()
    # Train the model with the new callback
    model.fit(
        train_images,
        train_labels,
        epochs=10,
        batch_size=32,
        validation_data=(test_images, test_labels),
        verbose=0
    )

    shutil.rmtree('tmp/checkpoints', ignore_errors=True)

    # Save the weights
    model.save_weights('tmp/checkpoints/my_checkpoint')

    # Create a new model instance
    new_model = create_model()

    # Restore the weights
    new_model.load_weights('tmp/checkpoints/my_checkpoint')

    # Evaluate the model
    loss, acc = model.evaluate(test_images, test_labels, verbose=0)
    logger.info("Original model's accuracy: {:5.2f}%".format(100 * acc))
    loss, acc = new_model.evaluate(test_images, test_labels, verbose=0)
    logger.info("Restored model's accuracy: {:5.2f}%".format(100 * acc))


args.step = auto_increment(args.step, args.all)
### Step #7 - Save the entire model: SavedModel format
if args.step == 7:
    print("\n### Step #7 - Save the entire model: SavedModel format")

    shutil.rmtree('tmp/saved_model', ignore_errors=True)

    model = create_model()
    model.fit(train_images, train_labels, epochs=5, verbose=0)
    model.save('tmp/saved_model/my_model')
    logger.info('ls -l tmp/saved_model/my_model:')
    print(*os.listdir('tmp/saved_model/my_model'), sep="\n")
    print('')

    new_model = tf.keras.models.load_model('tmp/saved_model/my_model')
    new_model.summary()
    loss, acc = new_model.evaluate(test_images, test_labels, verbose=0)
    logger.info("Restored model's accuracy: {:5.2f}%".format(100 * acc))
    logger.info(new_model.predict(test_images).shape)


args.step = auto_increment(args.step, args.all)
### Step #8 - Save the entire model: HDF5 format
if args.step == 8:
    print("\n### Step #8 - Save the entire model: HDF5 format")

    shutil.rmtree('tmp/saved_model', ignore_errors=True)
    model = create_model()
    model.fit(train_images, train_labels, epochs=5, verbose=0)

    # The '.h5' extension indicates that the model should be saved to HDF5.
    model.save('tmp/saved_model/my_model.h5')
    new_model = tf.keras.models.load_model('tmp/saved_model/my_model.h5')
    new_model.summary()
    loss, acc = new_model.evaluate(test_images, test_labels, verbose=0)
    logger.info("Restored model's accuracy: {:5.2f}%".format(100 * acc))


args.step = auto_increment(args.step, args.all)
### Step #9 - Save the entire model: Saving custom object
if args.step == 9:
    print("\n### Step #9 - Save the entire model: Saving custom object")
    pass


### End of File
if args.plot:
    plt.show()
debug()

