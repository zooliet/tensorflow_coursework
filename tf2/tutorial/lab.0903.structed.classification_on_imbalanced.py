#!/usr/bin/env python

import sys
sys.path.append('./')
sys.path.append('../../')

from lab_utils import (
    tf, os, np, plt, logger, ap, BooleanAction,
    debug, toc, auto_increment
)

ap.add_argument('--epochs', type=int, default=100, help='number of epochs: 100*')
ap.add_argument('--batch', type=int, default=2048, help='batch size: 2048*')
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
import tempfile
import pandas as pd
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Softmax


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - Data processing and exploration: Download the Kaggle Credit Card Fraud data set
if args.step >= 1: 
    print("\n### Step #1 - Data processing and exploration: Download the Kaggle Credit Card Fraud data set")

    filepath = f'{os.getenv("HOME")}/.keras/datasets/creditcard.csv'
    if not os.path.exists(filepath):
        filepath = tf.keras.utils.get_file(
            'creditcard.csv',
            'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'
        )

    raw_df = pd.read_csv(filepath)

    if args.step == 1:
        logger.info('raw_df')
        print(raw_df.head())
        print(
            raw_df[
                ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V26', 'V27', 'V28', 'Amount', 'Class']
            ].describe()
        )


args.step = auto_increment(args.step, args.all)
### Step #2 - Data processing and exploration: Examine the class label imbalance
if args.step >= 2: 
    print("\n### Step #2 - Data processing and exploration: Examine the class label imbalance")

    neg, pos = np.bincount(raw_df['Class'])
    total = neg + pos

    if args.step == 2:
        logger.info(f'Total: {total} examples')
        logger.info(f'Positive: {pos} ({100*pos/total:.2f} of total)')


args.step = auto_increment(args.step, args.all)
### Step #3 - Data processing and exploration: Clean, split and normalize the data
if args.step >= 3: 
    print("\n### Step #3 - Data processing and exploration: Clean, split and normalize the data")

    cleaned_df = raw_df.copy()

    # You don't want the `Time` column.
    cleaned_df.pop('Time')

    # The `Amount` column covers a huge range. Convert to log-space.
    eps = 0.001 # 0 => 0.1¢
    cleaned_df['Log Ammount'] = np.log(cleaned_df.pop('Amount')+eps)

    # Use a utility from sklearn to split and shuffle your dataset.
    train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
    train_df, val_df = train_test_split(train_df, test_size=0.2)

    # Form np arrays of labels and features.
    train_labels = np.array(train_df.pop('Class'))
    bool_train_labels = train_labels != 0
    val_labels = np.array(val_df.pop('Class'))
    test_labels = np.array(test_df.pop('Class'))

    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)

    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    train_features = np.clip(train_features, -5, 5)
    val_features = np.clip(val_features, -5, 5)
    test_features = np.clip(test_features, -5, 5)

    if args.step == 3:
        logger.info('Training labels shape: {}'.format(train_labels.shape))
        logger.info('Validation labels shape: {}'.format(val_labels.shape))
        logger.info('Test labels shape: {}'.format(test_labels.shape))

        logger.info('Training features shape: {}'.format(train_features.shape))
        logger.info('Validation features shape: {}'.format(val_features.shape))
        logger.info('Test features shape: {}'.format(test_features.shape))


args.step = auto_increment(args.step, args.all)
### Step #4 - Data processing and exploration: Look at the data distribution
if args.step == 4: 
    print("\n### Step #4 - Data processing and exploration: Look at the data distribution")

    if args.plot:
        pos_df = pd.DataFrame(train_features[bool_train_labels], columns=train_df.columns)
        neg_df = pd.DataFrame(train_features[~bool_train_labels], columns=train_df.columns)

        sns.jointplot(pos_df['V5'], pos_df['V6'], kind='hex', xlim=(-5,5), ylim=(-5,5))
        plt.suptitle("Positive distribution")

        sns.jointplot(neg_df['V5'], neg_df['V6'], kind='hex', xlim=(-5,5), ylim=(-5,5))
        _ = plt.suptitle("Negative distribution")
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #5 - Define the model and metrics
if args.step >= 5: 
    print("\n### Step #5 - Define the model and metrics")

    METRICS = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]

    def make_model(metrics=METRICS, output_bias=None):
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)

        model = Sequential([
            Dense(16, activation='relu', input_shape=(train_features.shape[-1],)),
            Dropout(0.5),
            Dense(1, activation='sigmoid', bias_initializer=output_bias),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=metrics
        )

        return model

    EPOCHS = args.epochs # 100
    BATCH_SIZE = args.batch # 2048

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_prc',
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True
    )


args.step = auto_increment(args.step, args.all)
### Step #6 - Define the model and metrics: Understanding useful metrics
if args.step == 6: 
    print("\n### Step #6 - Define the model and metrics: Understanding useful metrics")

    __doc__='''
    Notice that there are a few metrics defined above that can be computed by
    the model that will be helpful when evaluating the performance.

    - False negatives and false positives are samples that were incorrectly
      classified
    - True negatives and true positives are samples that were correctly
      classified
    - Accuracy is the percentage of examples correctly classified (True/Total) 
    - Precision is the percentage of predicted positives that were correctly
      classified (True pos/(True pos+False pos)) 
    - Recall is the percentage of actual positives that were correctly
      classified (True pos/(True pos+False neg)) 
    - AUC refers to the Area Under the Curve of a Receiver Operating
      Characteristic curve (ROC-AUC). This metric is equal to the probability
      that a classifier will rank a random positive sample higher than a random
      negative sample.
    - AUPRC refers to Area Under the Curve of the Precision-Recall Curve. This
      metric computes precision-recall pairs for different probability
      thresholds.
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #7 - Baseline model: Build the model
if args.step in  [7, 8, 9]: 
    print("\n### Step #7 - Baseline model: Build the model")

    # EPOCHS = args.epochs # 100
    # BATCH_SIZE = args.batch # 2048
    #
    # early_stopping = tf.keras.callbacks.EarlyStopping(
    #     monitor='val_prc',
    #     verbose=1,
    #     patience=10,
    #     mode='max',
    #     restore_best_weights=True
    # )

    model = make_model()
    # Test run the model
    preds = model.predict(train_features[:10])

    if args.step == 7:
        model.summary()
        print()

        logger.info('model.predict(train_features[:10]):')
        print(preds)


args.step = auto_increment(args.step, args.all)
### Step #8 - Baseline model: Set the correct initial bias
if args.step in [8, 9]: 
    print("\n### Step #8 - Baseline model: Set the correct initial bias")

    results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
    initial_bias = np.log([pos/neg])
    model = make_model(output_bias=initial_bias)
    preds = model.predict(train_features[:10])
    results_initial_bias = model.evaluate(
        train_features, train_labels, batch_size=BATCH_SIZE, verbose=0
    )

    if args.step == 8:
        logger.info("Loss: {:0.4f}".format(results[0]))
        logger.info(f'initial bias: {initial_bias}')
        logger.info('model.predict() with initial bias:')
        print(preds, '\n')
        logger.info("Loss with initial bias: {:0.4f}".format(results_initial_bias[0]))


args.step = auto_increment(args.step, args.all)
### Step #9 - Baseline model: Checkpoint the initial weights
if args.step == 9: 
    print("\n### Step #9 - Baseline model: Checkpoint the initial weights")

    # initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
    initial_weights = os.path.join('tmp/tf2_t0903/', 'initial_weights')
    model.save_weights(initial_weights)


args.step = auto_increment(args.step, args.all)
### Step #10 - Baseline model: Confirm that the bias fix helps
if args.step == 10: 
    print("\n### Step #10 - Baseline model: Confirm that the bias fix helps")

    initial_weights = os.path.join('tmp/tf2_t0903/', 'initial_weights')
    model = make_model()
    model.load_weights(initial_weights)
    model.layers[-1].bias.assign([0.0])
    zero_bias_history = model.fit(
        train_features,
        train_labels,
        batch_size=args.batch, # 2048
        epochs=20,
        validation_data=(val_features, val_labels), 
        verbose=0
    ) 

    model = make_model()
    model.load_weights(initial_weights)
    careful_bias_history = model.fit(
        train_features,
        train_labels,
        batch_size=args.batch, # 2048
        epochs=20,
        validation_data=(val_features, val_labels), 
        verbose=0
    )

    def plot_loss(history, label, n):
        # Use a log scale on y-axis to show the wide range of values.
        plt.semilogy(
            history.epoch, history.history['loss'], color=colors[n], label='Train ' + label
        )
        plt.semilogy(
            history.epoch, history.history['val_loss'], 
            color=colors[n], label='Val ' + label, linestyle="--"
        )
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

    if args.plot:
        plt.figure()
        plot_loss(zero_bias_history, "Zero Bias", 0)
        plot_loss(careful_bias_history, "Careful Bias", 1)
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #11 - Baseline model: Train the model
if args.step in [11, 12, 13, 14, 15]: 
    print("\n### Step #11 - Baseline model: Train the model")

    initial_weights = os.path.join('tmp/tf2_t0903/', 'initial_weights')
    model = make_model()
    model.load_weights(initial_weights)
    baseline_history = model.fit(
        train_features,
        train_labels,
        batch_size=args.batch, # 2048, BATCH_SIZE,
        epochs=args.epochs, # 100, EPOCHS,
        callbacks=[early_stopping],
        validation_data=(val_features, val_labels),
        verbose=2
    )


args.step = auto_increment(args.step, args.all)
### Step #12 - Baseline model: Check training history
if args.step == 12: 
    print("\n### Step #12 - Baseline model: Check training history")

    def plot_metrics(history):
        metrics = ['loss', 'prc', 'precision', 'recall']
        for n, metric in enumerate(metrics):
            name = metric.replace("_"," ").capitalize()
            plt.subplot(2,2,n+1)
            plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
            plt.plot(
                history.epoch, history.history['val_'+metric],
                color=colors[0], linestyle="--", label='Val'
            )
            plt.xlabel('Epoch')
            plt.ylabel(name)
            if metric == 'loss':
              plt.ylim([0, plt.ylim()[1]])
            elif metric == 'auc':
              plt.ylim([0.8,1])
            else:
              plt.ylim([0,1])

            plt.legend()

    if args.plot:
        plt.figure()
        plot_metrics(baseline_history)
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #13 - Baseline model: Evaluate metrics
if args.step >= 13: 
    print("\n### Step #13 - Baseline model: Evaluate metrics")

    train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)
    test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)

    def plot_cm(labels, predictions, p=0.5):
        cm = confusion_matrix(labels, predictions > p)
        plt.figure(figsize=(5,5))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title('Confusion matrix @{:.2f}'.format(p))
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.show(block=False)

        print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
        print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
        print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
        print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
        print('Total Fraudulent Transactions: ', np.sum(cm[1]))

    baseline_results = model.evaluate(
        test_features, test_labels, batch_size=BATCH_SIZE, verbose=0)

    if args.step == 13:
        for name, value in zip(model.metrics_names, baseline_results):
            print(name, ': ', value)
        print()

        if args.plot:
            plot_cm(test_labels, test_predictions_baseline)


args.step = auto_increment(args.step, args.all)
### Step #14 - Baseline model: Plot the ROC
if args.step >= 14: 
    print("\n### Step #14 - Baseline model: Plot the ROC")

    def plot_roc(name, labels, predictions, **kwargs):
        fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

        plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
        plt.xlabel('False positives [%]')
        plt.ylabel('True positives [%]')
        plt.xlim([-0.5,20])
        plt.ylim([80,100.5])
        plt.grid(True)
        ax = plt.gca()
        ax.set_aspect('equal')

    if args.step == 14 and args.plot:
        plt.figure()
        plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
        plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
        plt.legend(loc='lower right')
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #15 - Baseline model: Plot the AUPRC
if args.step >= 15: 
    print("\n### Step #15 - Baseline model: Plot the AUPRC")

    def plot_prc(name, labels, predictions, **kwargs):
        precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)

        plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True)
        ax = plt.gca()
        ax.set_aspect('equal')

    if args.step == 15 and args.plot:
        plt.figure()
        plot_prc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
        plot_prc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
        plt.legend(loc='lower right')
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #16 - Class weights: Calculate class weights
if args.step >= 16: 
    print("\n### Step #16 - Class weights: Calculate class weights")


args.step = auto_increment(args.step, args.all)
### Step #17 - Class weights: Train a model with class weights
if args.step >= 17: 
    print("\n### Step #17 - Class weights: Train a model with class weights")


args.step = auto_increment(args.step, args.all)
### Step #18 - Class weights: Check training history
if args.step >= 18: 
    print("\n### Step #18 - Class weights: Check training history")


args.step = auto_increment(args.step, args.all)
### Step #19 - Class weights: Evaluate metrics
if args.step >= 19: 
    print("\n### Step #19 - Class weights: Evaluate metrics")


args.step = auto_increment(args.step, args.all)
### Step #20 - Class weights: Plot the ROC
if args.step >= 20: 
    print("\n### Step #20 - Class weights: Plot the ROC")


args.step = auto_increment(args.step, args.all)
### Step #21 - Class weights: Plot the AUPRC
if args.step >= 21: 
    print("\n### Step #21 - Class weights: Plot the AUPRC")


args.step = auto_increment(args.step, args.all)
### Step #22 - Oversampling: Oversample the minority class
if args.step >= 22: 
    print("\n### Step #22 - Oversampling: Oversample the minority class")


args.step = auto_increment(args.step, args.all)
### Step #23 - Oversampling: Train on the oversampled data
if args.step >= 23: 
    print("\n### Step #23 - Oversampling: Oversample the minority class: Train on the oversampled data")



args.step = auto_increment(args.step, args.all)
### Step #24 - Oversampling: Check training history
if args.step >= 24: 
    print("\n### Step #24 - Oversampling: Check training history")


args.step = auto_increment(args.step, args.all)
### Step #25 - Oversampling: Re-train
if args.step >= 25: 
    print("\n### Step #25 - Oversampling: Re-train")


args.step = auto_increment(args.step, args.all)
### Step #26 - Oversampling: Re-check training history
if args.step >= 26: 
    print("\n### Step #26 - Oversampling: Re-check training history")


args.step = auto_increment(args.step, args.all)
### Step #27 - Oversampling: Evaluate metrics
if args.step >= 27: 
    print("\n### Step #27 - Oversampling: Evaluate metrics")


args.step = auto_increment(args.step, args.all)
### Step #28 - Oversampling: Plot the ROC
if args.step >= 28: 
    print("\n### Step #28 - Oversampling: Plot the ROC")


args.step = auto_increment(args.step, args.all)
### Step #29 - Oversampling: Plot the AUPRC
if args.step >= 29: 
    print("\n### Step #29 - Oversampling: Plot the AUPRC")


args.step = auto_increment(args.step, args.all)
### Step #30 - Applying this tutorial to your problem
if args.step >= 30: 
    print("\n### Step #30 - Applying this tutorial to your problem")




### End of File
print()
if args.plot:
    plt.show()
debug()


