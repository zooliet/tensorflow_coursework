#!/usr/bin/env python


import sys
sys.path.append('./')
sys.path.append('../')

from lab_utils import (
    os, np, plt, logger, ap, BooleanAction,
    debug, toc
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

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("\n#################################################")
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense 

### Step #0 - TOC
if args.step == 0:
    toc(__file__)


### Step #1 - Import and parse the training dataset: Download the dataset
if args.step >= 1:
    print("\n### Step #1 - Import and parse the training dataset: Download the dataset")

    train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
    train_dataset_fp = tf.keras.utils.get_file(
        fname=os.path.basename(train_dataset_url),
        origin=train_dataset_url
    )

    logger.info("Local copy of the dataset file: {}".format(train_dataset_fp))


### Step #2 - Import and parse the training dataset: Inspect the data
if args.step >= 2:
    print("\n### Step #2 - Import and parse the training dataset: Inspect the data")

    if args.step == 2:
        os.system(f'head -n5 {train_dataset_fp}')

    # column order in CSV file
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

    feature_names = column_names[:-1]
    label_name = column_names[-1]
    logger.info("Features: {}".format(feature_names))
    logger.info("Label: {}".format(label_name))

    class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']


### Step #3 - Import and parse the training dataset: Create a tf.data.Dataset
if args.step >= 3:
    print("\n### Step #3 - Import and parse the training dataset: Create a tf.data.Dataset")

    batch_size = 16

    train_dataset = tf.data.experimental.make_csv_dataset(
        train_dataset_fp,
        batch_size,
        column_names=column_names,
        label_name=label_name,
        num_epochs=1,
        shuffle=False
        # shuffle=True, # default
        # shuffle_buffer_size=10000 # default
    )

    if args.step == 3:
        features, labels = next(iter(train_dataset))
        for feature_name, values in features.items():
            print(f'{feature_name:16s}: {values.numpy()}')
        print('{:14s}: {}\n'.format('labels', labels.numpy()))

        if args.step == 3 and args.plot:
            plt.scatter(
                features['petal_length'],
                features['sepal_length'],
                c=labels,
                cmap='viridis'
            )

            plt.xlabel("Petal length")
            plt.ylabel("Sepal length")
            plt.show(block=False)

    def pack_features_vector(features, labels):
        """Pack the features into a single array."""
        # (4, 16) => (16, 4)
        features = tf.stack(list(features.values()), axis=1)
        return features, labels

    train_dataset = train_dataset.map(pack_features_vector)

    if args.step == 3:
        logger.info('multi-inputs(single feature) => single input(multi-features):')
        features, labels = next(iter(train_dataset))
        print(features)


### Step #4 - Select the type of model: Create a model using Keras
if args.step >= 4:
    print("\n### Step #4 - Select the type of model: Create a model using Keras")

    model = Sequential([
        Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
        Dense(10, activation=tf.nn.relu),
        Dense(3)
    ])

    if args.step == 4:
        model.summary()


### Step #5 - Select the type of model: Using the model
if args.step >= 5:
    print("\n### Step #5 - Select the type of model: Using the model")

    features, labels = next(iter(train_dataset))
    predictions = model(features)

    if args.step == 5:
        logger.info('predictions before training:')
        print(predictions[:5])


### Step #6 - Train the model: Define the loss and gradient function
if args.step >= 6:
    print("\n### Step #6 - Train the model: Define the loss and gradient function")

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def loss(model, x, y, training):
        # training=training is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        y_ = model(x, training=training)
        return loss_object(y_true=y, y_pred=y_)

    if args.step == 6:
        l = loss(model, features, labels, training=False)
        logger.info("Loss test: {:.2f}".format(l))

    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)


### Step #7 - Train the model: Create an optimizer
if args.step >= 7:
    print("\n### Step #7 - Train the model: Create an optimizer")

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    if args.step == 7:
        loss_value, grads = grad(model, features, labels)
        print("Step: {}, Initial Loss: {}".format(
            optimizer.iterations.numpy(), loss_value.numpy())
        )

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print("Step: {},         Loss: {}".format(
            optimizer.iterations.numpy(), loss(model, features, labels, training=True).numpy())
        ) 

        loss_value, grads = grad(model, features, labels)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print("Step: {},         Loss: {}".format(
            optimizer.iterations.numpy(), loss(model, features, labels, training=True).numpy())
        ) 


### Step #8 - Train the model: Training loop
if args.step >= 8:
    print("\n### Step #8 - Train the model: Training loop")

    ## Note: Rerunning this cell uses the same model variables
    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    num_epochs = 201

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Training loop - using batches of 32
        for x, y in train_dataset:
            # Optimize the model
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            epoch_accuracy.update_state(y, model(x, training=True))

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(
                epoch, epoch_loss_avg.result(), epoch_accuracy.result())
        )

        # 원래는 해야함
        # epoch_loss_avg.reset_states()
        # epoch_accuracy.reset_states()


### Step #9 - Train the model: Visualize the loss function over time
if args.step >= 9:
    print("\n### Step #9 - Train the model: Visualize the loss function over time")

    if args.step == 9 and args.plot:
        fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
        fig.suptitle('Training Metrics')

        axes[0].set_ylabel("Loss", fontsize=14)
        axes[0].plot(train_loss_results)

        axes[1].set_ylabel("Accuracy", fontsize=14)
        axes[1].set_xlabel("Epoch", fontsize=14)
        axes[1].plot(train_accuracy_results)
        plt.show(block=False)


### Step #10 - Evaluate the model's effectiveness: Setup the test dataset
if args.step >= 10:
    print("\n### Step #10 - Evaluate the model's effectiveness: Setup the test dataset")

    test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
    test_fp = tf.keras.utils.get_file(
        fname=os.path.basename(test_url),
        origin=test_url
    )

    test_dataset = tf.data.experimental.make_csv_dataset(
        test_fp,
        batch_size,
        column_names=column_names,
        label_name='species',
        num_epochs=1,
        shuffle=False
    )

    test_dataset = test_dataset.map(pack_features_vector)


### Step #11 - Evaluate the model's effectiveness: Evaluate the model on the test dataset
if args.step >= 11:
    print("\n### Step #11 - Evaluate the model's effectiveness: Evaluate the model on the test dataset")

    test_accuracy = tf.keras.metrics.Accuracy()

    for (x, y) in test_dataset:
        # training=False is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        logits = model(x, training=False)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy.update_state(prediction, y)

    logger.info("Test set accuracy: {:.3%}".format(test_accuracy.result()))

    if args.step == 11:
        print(tf.stack([y, prediction], axis=1))


### Step #12 - Use the trained model to make predictions
if args.step >= 12:
    print("\n### Step #12 - Use the trained model to make predictions")

    predict_dataset = tf.convert_to_tensor([
        [5.1, 3.3, 1.7, 0.5,],
        [5.9, 3.0, 4.2, 1.5,],
        [6.9, 3.1, 5.4, 2.1]
    ])

    # training=False is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(predict_dataset, training=False)

    for i, logits in enumerate(predictions):
        class_idx = tf.argmax(logits).numpy()
        p = tf.nn.softmax(logits)[class_idx]
        name = class_names[class_idx]
        logger.info("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))


### End of File
if args.plot:
    plt.show()
debug()


