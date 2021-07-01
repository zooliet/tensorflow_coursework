#!/usr/bin/env python

import sys
sys.path.append('./')
sys.path.append('../../')

from lab_utils import (
    tf, os, np, plt, logger, ap, BooleanAction,
    debug, toc, auto_increment
)

ap.add_argument('--epochs', type=int, default=2, help='number of epochs: 2*')
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

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.layers import Concatenate, Layer, Reshape
from tensorflow.keras.layers import Conv1D, Conv2D, Conv2DTranspose 
from tensorflow.keras.layers import GlobalMaxPooling1D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, UpSampling2D
from tensorflow.keras.layers import Embedding, LSTM


### TOC
if args.step == 0:
    toc(__file__)


if args.step or args.all:
    if not os.path.exists('tmp/tf2_g0203/'):
        os.mkdir('tmp/tf2_g0203/')

    # download the dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # preprocess the data (these are NumPy arrays)
    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255

    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # reserve 10,000 samples for validation
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    # model builder
    def get_uncompiled_model():
        inputs = Input(shape=(784,), name="digits")
        x = Dense(64, activation="relu", name="dense_1")(inputs)
        x = Dense(64, activation="relu", name="dense_2")(x)
        outputs = Dense(10, activation="softmax", name="predictions")(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def get_compiled_model():
        model = get_uncompiled_model()
        model.compile(
            optimizer="rmsprop",
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
        )
        return model


args.step = auto_increment(args.step, args.all)
### Step #1 - API overview: a first end-to-end example
if args.step == 1:
    print("\n### Step #1 - API overview: a first end-to-end example")

    inputs = Input(shape=(784,), name="digits")
    x = Dense(64, activation="relu", name="dense_1")(inputs)
    x = Dense(64, activation="relu", name="dense_2")(x)
    outputs = Dense(10, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),  # Optimizer
        # Loss function to minimize
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=args.epochs,
        # We pass some validation for monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(x_val, y_val),
        verbose=2
    )
    print()

    logger.info('history.history:')
    for key, values in history.history.items():
        values = [round(v, 2) for v in values]
        print(f'{key}: {values}')
    print()

    # Evaluate the model on the test data using `evaluate`
    results = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
    logger.info("test loss: {:.2f}".format(results[0]))
    logger.info("test acc: {:.2f}".format(results[1]))

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    predictions = model.predict(x_test[:3])
    logger.info("predictions {}:".format(predictions.shape))
    print(predictions)


args.step = auto_increment(args.step, args.all)
### Step #2 - The compile() method: specifying a loss, metrics, and an optimizer
if args.step == 2:
    print("\n### Step #2 - The compile() method: specifying a loss, metrics, and an optimizer")

    __doc__='''
    To train a model with fit(), you need to specify a loss function, an
    optimizer, and optionally, some metrics to monitor. You pass these to the
    model as arguments to the compile() method

    The metrics argument should be a list -- your model can have any number of
    metrics.

    If your model has multiple outputs, you can specify different losses and
    metrics for each output, and you can modulate the contribution of each
    output to the total loss of the model.
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #3 - The compile() method: Custom losses
if args.step == 3:
    print("\n### Step #3 - The compile() method: Custom losses")

    def custom_mean_squared_error(y_true, y_pred):
        return tf.math.reduce_mean(tf.square(y_true - y_pred))

    model = get_uncompiled_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(), 
        loss=custom_mean_squared_error
    )

    # We need to one-hot encode the labels to use MSE
    y_train_one_hot = tf.one_hot(y_train, depth=10)
    history = model.fit(
        x_train, 
        y_train_one_hot, 
        epochs=args.epochs,
        batch_size=64, 
        verbose=2
    )

    logger.info('history.history:')
    for key, values in history.history.items():
        values = [round(v, 2) for v in values]
        print(f'{key}: {values}')
    print()

    # using class
    class CustomMSE(tf.keras.losses.Loss):
        def __init__(self, regularization_factor=0.1, name="custom_mse"):
            super().__init__(name=name)
            self.regularization_factor = regularization_factor

        def call(self, y_true, y_pred):
            mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
            reg = tf.math.reduce_mean(tf.square(0.5 - y_pred))
            return mse + reg * self.regularization_factor

    model = get_uncompiled_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(), 
        loss=CustomMSE()
    )

    y_train_one_hot = tf.one_hot(y_train, depth=10)
    history = model.fit(
        x_train, 
        y_train_one_hot, 
        epochs=args.epochs,
        batch_size=64, 
        verbose=2
    )
    for key, values in history.history.items():
        values = [round(v, 2) for v in values]
        print(f'{key}: {values}')


args.step = auto_increment(args.step, args.all)
### Step #4 - The compile() method: Custom metrics
if args.step == 4:
    print("\n### Step #4 - The compile() method: Custom metrics")

    class CategoricalTruePositives(tf.keras.metrics.Metric):
        def __init__(self, name="categorical_true_positives", **kwargs):
            super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
            self.true_positives = self.add_weight(name="ctp", initializer="zeros")

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
            values = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
            values = tf.cast(values, "float32")
            if sample_weight is not None:
                sample_weight = tf.cast(sample_weight, "float32")
                values = tf.multiply(values, sample_weight)
            self.true_positives.assign_add(tf.reduce_sum(values))
            # tf.print(self.true_positives)

        def result(self):
            return self.true_positives

        # The state of the metric will be reset at the start of each epoch.
        def reset_states(self):
            self.true_positives.assign(0.0)

    model = get_uncompiled_model()
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[CategoricalTruePositives()],
    )
    history = model.fit(
        x_train, 
        y_train, 
        epochs=args.epochs,
        batch_size=64, 
        verbose=2
    )
    for key, values in history.history.items():
        values = [round(v, 2) for v in values]
        print(f'{key}: {values}')


args.step = auto_increment(args.step, args.all)
### Step #5 - The compile() method: Handling losses and metrics that don't fit the standard signature
if args.step == 5:
    print("\n### Step #5 - The compile() method: Handling losses and metrics that don't fit the standard signature")

    __doc__='''
    The overwhelming majority of losses and metrics can be computed from y_true
    and y_pred, where y_pred is an output of your model -- but not all of them.
    For instance, a regularization loss may only require the activation of a
    layer (there are no targets in this case), and this activation may not be a
    model output.

    In such cases, you can call self.add_loss(loss_value) from inside the call
    method of a custom layer. Losses added in this way get added to the "main"
    loss during training (the one passed to compile()). Here's a simple example
    that adds activity regularization (note that activity regularization is
    built-in in all Keras layers -- this layer is just for the sake of
    providing a concrete example).
    '''

    class ActivityRegularizationLayer(Layer):
        def call(self, inputs):
            self.add_loss(tf.reduce_sum(inputs) * 0.1)
            return inputs  # Pass-through layer.

    inputs = Input(shape=(784,), name="digits")
    x = Dense(64, activation="relu", name="dense_1")(inputs)

    # Insert activity regularization as a layer
    x = ActivityRegularizationLayer()(x)

    x = Dense(64, activation="relu", name="dense_2")(x)
    outputs = Dense(10, name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )

    # The displayed loss will be much higher than before
    # due to the regularization component.
    history = model.fit(
        x_train, 
        y_train, 
        epochs=args.epochs,
        batch_size=64, 
        verbose=2
    )
    for key, values in history.history.items():
        values = [round(v, 2) for v in values]
        print(f'{key}: {values}')


args.step = auto_increment(args.step, args.all)
### Step #6 - The compile() method: Automatically setting apart a validation holdout set
if args.step == 6:
    print("\n### Step #6 - The compile() method: Automatically setting apart a validation holdout set")

    __doc__='''
    The argument validation_split allows you to automatically reserve part of
    your training data for validation. The argument value represents the
    fraction of the data to be reserved for validation, so it should be set to
    a number higher than 0 and lower than 1.  For instance,
    validation_split=0.2 means "use 20% of the data for validation". 
    
    The way the validation is computed is by taking the last x% samples of the
    arrays received by the fit() call, before any shuffling.
    '''
    print(__doc__)

    model = get_compiled_model()
    history = model.fit(
        x_train, y_train, 
        epochs=args.epochs,
        batch_size=64, 
        validation_split=0.2, 
        verbose=2
    )


args.step = auto_increment(args.step, args.all)
### Step #7 - Training & evaluation from tf.data Datasets
if args.step == 7:
    print("\n### Step #7 - Training & evaluation from tf.data Datasets")

    model = get_compiled_model()

    # First, let's create a training Dataset instance.
    # For the sake of our example, we'll use the same MNIST data as before.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # Shuffle and slice the dataset.
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

    # Prepare the validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(64)

    # Now we get a test dataset.
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(64)

    # Since the dataset already takes care of batching,
    # we don't pass a `batch_size` argument.
    history = model.fit(
        train_dataset, 
        epochs=args.epochs, 
        validation_data=val_dataset,
        verbose=2
    )
    print()

    for key, values in history.history.items():
        values = [round(v, 2) for v in values]
        print(f'{key}: {values}')
    print()

    # You can also evaluate or predict on a dataset.
    result = model.evaluate(test_dataset, verbose=0)
    for name, value in dict(zip(model.metrics_names, result)).items():
        logger.info(f'{name}: {value:.2f}')
    

args.step = auto_increment(args.step, args.all)
### Step #8 - Training & evaluation from tf.data Datasets: using steps_per_epoch
if args.step == 8:
    print("\n### Step #8 - Training & evaluation from tf.data Datasets: using steps_per_epoch")

    __doc__='''
    steps_per_epcoh specifies how many training steps the model should run
    using this Dataset before moving on to the next epoch. If you do this, the
    dataset is not reset at the end of each epoch, instead we just keep drawing
    the next batches.  The dataset will eventually run out of data (unless it
    is an infinitely-looping dataset).
    '''
    print(__doc__)

    model = get_compiled_model()

    # Prepare the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

    # Only use the 100 batches per epoch (that's 64 * 100 samples)
    history=model.fit(
        train_dataset, 
        epochs=args.epochs, # 2 is ok, but 10 needs more data
        steps_per_epoch=100, 
        verbose=1
    )


args.step = auto_increment(args.step, args.all)
### Step #9 - Training & evaluation from tf.data Datasets: Using a validation dataset
if args.step == 9:
    print("\n### Step #9 - Training & evaluation from tf.data Datasets: Using a validation dataset")

    model = get_compiled_model()

    # Prepare the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

    # Prepare the validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(64)

    history = model.fit(
        train_dataset, 
        epochs=args.epochs, 
        validation_data=val_dataset,
        verbose=2
    )


args.step = auto_increment(args.step, args.all)
### Step #10 - Using a keras.utils.Sequence object as input
if args.step == 10:
    print("\n### Step #10 - Using a keras.utils.Sequence object as input")

    __doc__='''
    Besides NumPy arrays, eager tensors, and TensorFlow Datasets, it's possible
    to train a Keras model using Pandas dataframes, or from Python generators
    that yield batches of data & labels.

    In particular, the keras.utils.Sequence class offers a simple interface to
    build Python data generators that are multiprocessing-aware and can be
    shuffled.

    In general, we recommend that you use:
    - NumPy input data if your data is small and fits in memory
    - Dataset objects if you have large datasets and you need to do distributed
      training
    - Sequence objects if you have large datasets and you need to do a lot of
      custom Python-side processing that cannot be done in TensorFlow (e.g. if
      you rely on external libraries for data loading or preprocessing).
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #11 - Using sample weighting and class weighting: Class weights
if args.step == 11:
    print("\n### Step #11 - Using sample weighting and class weighting: Class weights")
    
    __doc__='''
    This can be used to balance classes without resampling, or to train a model
    that gives more importance to a particular class.  For instance, if class
    "0" is half as represented as class "1" in your data, you could use
    Model.fit(..., class_weight={0: 1., 1: 0.5}).
    '''
    print(__doc__)

    class_weight = {
        0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0,
        # Set weight "2" for class "5",
        # making this class 2x more important
        5: 2.0,
        6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0,
    }

    model = get_compiled_model()
    history = model.fit(
        x_train, y_train, 
        class_weight=class_weight, 
        epochs=args.epochs, 
        batch_size=64, 
        verbose=2
    )


args.step = auto_increment(args.step, args.all)
### Step #12 - Using sample weighting and class weighting: Sample weights
if args.step == 12:
    print("\n### Step #12 - Using sample weighting and class weighting: Sample weights")
    pass


args.step = auto_increment(args.step, args.all)
### Step #13 - Passing data to multi-input, multi-output models
if args.step == 13:
    print("\n### Step #13 - Passing data to multi-input, multi-output models")

    __doc__='''
    Consider the following model, which has an image input of shape (32, 32, 3)
    (that's (height, width, channels)) and a time series input of shape (None,
    10) (that's (timesteps, features)).  Our model will have two outputs
    computed from the combination of these inputs: a "score" (of shape (1,))
    and a probability distribution over five classes (of shape (5,)).
    '''
    print(__doc__)

    image_input = Input(shape=(32, 32, 3), name="img_input")
    timeseries_input = Input(shape=(None, 10), name="ts_input")

    x1 = Conv2D(3, 3)(image_input)
    x1 = GlobalMaxPooling2D()(x1)

    x2 = Conv1D(3, 3)(timeseries_input)
    x2 = GlobalMaxPooling1D()(x2)

    x = tf.keras.layers.concatenate([x1, x2])

    score_output = Dense(1, name="score_output")(x)
    class_output = Dense(5, name="class_output")(x)

    model = Model(
        inputs=[image_input, timeseries_input], 
        outputs=[score_output, class_output]
    )

    tf.keras.utils.plot_model(
        model, "tmp/tf2_g0203/multi_input_and_output_model.png", show_shapes=True
    )
    if args.plot:
        image = Image.open('tmp/tf2_g0203/multi_input_and_output_model.png')
        plt.figure()
        plt.imshow(image)
        plt.show(block=False)

    # model.compile(
    #     optimizer=tf.keras.optimizers.RMSprop(1e-3),
    #     loss=[tf.keras.losses.MeanSquaredError(), tf.keras.losses.CategoricalCrossentropy()],
    #     metrics=[
    #         [tf.keras.metrics.MeanAbsolutePercentageError(), tf.keras.metrics.MeanAbsoluteError()],
    #         [tf.keras.metrics.CategoricalAccuracy()],
    #     ],
    # )

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(1e-3),
        loss={
            "score_output": tf.keras.losses.MeanSquaredError(),
            "class_output": tf.keras.losses.CategoricalCrossentropy(),
        },
        metrics={
            "score_output": [
                tf.keras.metrics.MeanAbsolutePercentageError(),
                tf.keras.metrics.MeanAbsoluteError(),
            ],
            "class_output": [tf.keras.metrics.CategoricalAccuracy()],
        },
        loss_weights={"score_output": 2.0, "class_output": 1.0},
    )

    # Generate dummy NumPy data
    img_data = np.random.random_sample(size=(100, 32, 32, 3))
    ts_data = np.random.random_sample(size=(100, 20, 10))
    score_targets = np.random.random_sample(size=(100, 1))
    class_targets = np.random.random_sample(size=(100, 5))

    # # Fit on lists
    # model.fit(
    #     [img_data, ts_data],
    #     [score_targets, class_targets],
    #     batch_size=32,
    #     epochs=1
    # )

    # # alternatively, fit on dicts
    # history = model.fit(
    #     {"img_input": img_data, "ts_input": ts_data},
    #     {"score_output": score_targets, "class_output": class_targets},
    #     batch_size=32,
    #     epochs=1,
    # )

    # or
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {"img_input": img_data, "ts_input": ts_data},
        {"score_output": score_targets, "class_output": class_targets},
    ))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
    history = model.fit(
        train_dataset, 
        epochs=args.epochs,
        verbose=2
    )
    print()

    logger.info('history.history:')
    for key, values in history.history.items():
        values = [round(v, 2) for v in values]
        print(f'{key}: {values}')
    # logger.info(model.metrics_names)


args.step = auto_increment(args.step, args.all)
### Step #14 - Using callbacks
if args.step == 14:
    print("\n### Step #14 - Using callbacks")

    __doc__='''
    Callbacks in Keras are objects that are called at different points during
    training (at the start of an epoch, at the end of a batch, at the end of an
    epoch, etc.). They can be used to implement certain behaviors, such as:
    - Doing validation at different points during training (beyond the built-in
      per-epoch) 
    - Checkpointing the model at regular intervals or when it exceeds a certain
      threshold
    - Changing the learning rate of the model when training seems to be
      plateauing
    - Doing fine-tuning of the top layers when training seems to be plateauing
    - Sending email or instant message notifications when training ends or so 
    '''
    print(__doc__)

    model = get_compiled_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            monitor="val_loss",
            # "no longer improving" being defined as "no better than 1e-2 less"
            min_delta=1e-2,
            # "no longer improving" being further defined as "for at least 2 epochs"
            patience=2,
            verbose=1,
        )
    ]
    history = model.fit(
        x_train,
        y_train,
        epochs=args.epochs, # 20
        batch_size=64,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=2
    )


args.step = auto_increment(args.step, args.all)
### Step #15 - Checkpointing models
if args.step == 15:
    print("\n### Step #15 - Checkpointing models")

    model = get_compiled_model()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            # Path where to save the model
            # The two parameters below mean that we will overwrite
            # the current checkpoint if and only if
            # the `val_loss` score has improved.
            # The saved model name will include the current epoch.
            filepath="tmp/tf2_g0203/mymodel_{epoch}_{loss:.2f}",
            save_best_only=True,  # Only save a model if `val_loss` has improved.
            monitor="val_loss",
            verbose=1,
        )
    ]
    history = model.fit(
        x_train, y_train, 
        epochs=args.epochs, 
        batch_size=64, 
        callbacks=callbacks, 
        validation_split=0.2,
        verbose=2
    )


args.step = auto_increment(args.step, args.all)
### Step #16 - Checkingpoint models: restore
if args.step == 16:
    print("\n### Step #16 - Checkingpoint models: restore")

    # Prepare a directory to store all the checkpoints.
    checkpoint_dir = "tmp/tf2_g0203/ckpt"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    def make_or_restore_model():
        # Either restore the latest model, or create a fresh one
        # if there is no checkpoint available.
        checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            logger.info("Restoring from {}".format(latest_checkpoint))
            return tf.keras.models.load_model(latest_checkpoint)
        logger.info("Creating a new model")
        return get_compiled_model()

    model = make_or_restore_model()
    callbacks = [
        # This callback saves a SavedModel every 100 batches.
        # We include the training loss in the saved model name.
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + '/ckpt-loss_{loss:.2f}',
            save_freq=100
        )
    ]

    history = model.fit(
        x_train, y_train, 
        epochs=args.epochs, 
        callbacks=callbacks, 
        verbose=2
    )


args.step = auto_increment(args.step, args.all)
### Step #17 - Using learning rate schedules
if args.step == 17:
    print("\n### Step #17 - Using learning rate schedules")

    __doc__='''
    A common pattern when training deep learning models is to gradually reduce
    the learning as training progresses. This is generally known as "learning
    rate decay".  The learning decay schedule could be static (fixed in
    advance, as a function of the 

    current epoch or the current batch index), or dynamic (responding to the
    current behavior of the model, in particular the validation loss).

    Several built-in schedules are available: ExponentialDecay,
    PiecewiseConstantDecay, PolynomialDecay, and InverseTimeDecay.
    '''
    print(__doc__)

    initial_learning_rate = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, 
        decay_steps=100000, 
        decay_rate=0.96, 
        staircase=True
    )
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)


args.step = auto_increment(args.step, args.all)
### Step #18 - Using learning rate schedules: Using callbacks to implement a dynamic learning rate schedule
if args.step == 18:
    print("\n### Step #18 - Using learning rate schedules: Using callbacks to implement a dynamic learning rate schedule")

    __doc__='''
    A dynamic learning rate schedule (for instance, decreasing the learning
    rate when the validation loss is no longer improving) cannot be achieved
    with these schedule objects, since the optimizer does not have access to
    validation metrics.

    However, callbacks do have access to all metrics, including validation
    metrics! You can thus achieve this pattern by using a callback that
    modifies the current learning rate on the optimizer. In fact, this is even
    built-in as the ReduceLROnPlateau callback.
    '''
    print(__doc__)

    
args.step = auto_increment(args.step, args.all)
### Step #19 - Visualizing loss and metrics during training
if args.step == 19:
    print("\n### Step #19 - Visualizing loss and metrics during training")
    logger.info("https://www.tensorflow.org/tensorboard")


### End of File
print()
if args.plot:
    plt.show()
debug()
