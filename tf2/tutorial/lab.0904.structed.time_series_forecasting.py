#!/usr/bin/env python

import sys
sys.path.append('./')
sys.path.append('../../')

from lab_utils import (
    tf, os, np, plt, logger, ap, BooleanAction,
    debug, toc, auto_increment
)

ap.add_argument('--epochs', type=int, default=20, help='number of epochs: 20*')
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
import pandas as pd
import seaborn as sns
# from PIL import Image
# import pathlib

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Softmax
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - The weather dataset
if args.step >= 1: 
    print("\n### Step #1 - The weather dataset")

    zip_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=True
    )
    csv_path, _ = os.path.splitext(zip_path)

    df = pd.read_csv(csv_path)

    # for hourly predictions, start by sub-sampling the data from 10 minute intervals to 1h
    # slice [start:stop:step], starting from index 5 take every 6th record.
    df = df[5::6]

    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

    if args.step == 1:
        print(df.head())

        if args.plot:
            plot_cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)']
            plot_features = df[plot_cols]
            plot_features.index = date_time
            _ = plot_features.plot(subplots=True)

            plot_features = df[plot_cols][:480]
            plot_features.index = date_time[:480]
            _ = plot_features.plot(subplots=True)
            plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #2 - The weather dataset: Inspect and cleanup
if args.step >= 2: 
    print("\n### Step #2 - The weather dataset: Inspect and cleanup")

    if args.step == 2: print(df.describe().transpose(), '\n')

    # wind velocity
    wv = df['wv (m/s)']
    bad_wv = wv == -9999.0
    wv[bad_wv] = 0.0 # in-line replacement

    max_wv = df['max. wv (m/s)']
    bad_max_wv = max_wv == -9999.0
    max_wv[bad_max_wv] = 0.0 # in-line replacement

    if args.step == 2:
        # The above inplace edits are reflected in the DataFrame
        logger.info(f"df['wv (m/s)'].min(): {df['wv (m/s)'].min()}")


args.step = auto_increment(args.step, args.all)
### Step #3 - The weather dataset: Feature engineering
if args.step >= 3: 
    print("\n### Step #3 - The weather dataset: Feature engineering")

    ## wind
    if args.step == 3 and args.plot:
        plt.figure()
        plt.hist2d(df['wd (deg)'], df['wv (m/s)'], bins=(50, 50), vmax=400)
        plt.colorbar()
        plt.xlabel('Wind Direction [deg]')
        plt.ylabel('Wind Velocity [m/s]')    
        plt.show(block=False)

    wv = df.pop('wv (m/s)')
    max_wv = df.pop('max. wv (m/s)')

    # Convert to radians.
    wd_rad = df.pop('wd (deg)')*np.pi / 180

    # Calculate the wind x and y components.
    df['Wx'] = wv*np.cos(wd_rad)
    df['Wy'] = wv*np.sin(wd_rad)

    # Calculate the max wind x and y components.
    df['max Wx'] = max_wv*np.cos(wd_rad)
    df['max Wy'] = max_wv*np.sin(wd_rad)

    if args.step == 3 and args.plot:
        plt.figure()
        plt.hist2d(df['Wx'], df['Wy'], bins=(50, 50), vmax=400)
        plt.colorbar()
        plt.xlabel('Wind X [m/s]')
        plt.ylabel('Wind Y [m/s]')
        ax = plt.gca()
        ax.axis('tight')
        plt.show(block=False)

    ## time
    # Start by converting it to seconds
    timestamp_s = date_time.map(pd.Timestamp.timestamp)

    day = 24*60*60
    year = (365.2425)*day

    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    if args.step == 3 and args.plot:
        plt.figure()
        plt.plot(np.array(df['Day sin'])[:25])
        plt.plot(np.array(df['Day cos'])[:25])
        plt.xlabel('Time [h]')
        plt.title('Time of day signal')
        plt.show(block=False)

        fft = tf.signal.rfft(df['T (degC)'])
        f_per_dataset = np.arange(0, len(fft))

        n_samples_h = len(df['T (degC)'])
        hours_per_year = 24*365.2524
        years_per_dataset = n_samples_h/(hours_per_year)

        f_per_year = f_per_dataset/years_per_dataset
        plt.figure()
        plt.step(f_per_year, np.abs(fft))
        plt.xscale('log')
        plt.ylim(0, 400000)
        plt.xlim([0.1, max(plt.xlim())])
        plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
        _ = plt.xlabel('Frequency (log scale)')
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #4 - The weather dataset: Split the data
if args.step >= 4: 
    print("\n### Step #4 - The weather dataset: Split the data")

    column_indices = {name: i for i, name in enumerate(df.columns)}

    # (70%, 20%, 10%) split for the training, validation, and test sets
    # data is not being randomly shuffled before splitting
    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    num_features = df.shape[1]

    if args.step == 4:
        logger.info(f'train_df: {len(train_df)}')
        logger.info(f'val_df: {len(val_df)}')
        logger.info(f'test_df: {len(test_df)}')
        logger.info(f'num_features: {num_features}')


args.step = auto_increment(args.step, args.all)
### Step #5 - The weather dataset: Normalize the data
if args.step >= 5: 
    print("\n### Step #5 - The weather dataset: Normalize the data")

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    if args.step == 5 and args.plot:
        df_std = (df - train_mean) / train_std
        df_std = df_std.melt(var_name='Column', value_name='Normalized')
        plt.figure(figsize=(12, 6))
        ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
        _ = ax.set_xticklabels(df.keys(), rotation=90)
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #6 - Data windowing: Indexes and offsets
if args.step >= 6: 
    print("\n### Step #6 - Data windowing: Indexes and offsets")

    class WindowGenerator():
      def __init__(self, input_width, label_width, shift,
                   train_df=train_df, val_df=val_df, test_df=test_df,
                   label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
          self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}

        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

      def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    w1 = WindowGenerator(
        input_width=24, label_width=1, shift=24, label_columns=['T (degC)']
    )
    w2 = WindowGenerator(
        input_width=6, label_width=1, shift=1, label_columns=['T (degC)']
    )

    if args.step == 6:
        print(w1, '\n')
        print(w2)

        if args.plot:
            plt.figure()
            img = tf.io.read_file('supplement/tf2_t0904_01.png')
            img = tf.image.decode_png(img)
            plt.imshow(img)

            plt.figure()
            img = tf.io.read_file('supplement/tf2_t0904_02.png')
            img = tf.image.decode_png(img)
            plt.imshow(img)
            plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #7 - Data windowing: Split
if args.step >= 7: 
    print("\n### Step #7 - Data windowing: Split")

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    WindowGenerator.split_window = split_window

    example_window = tf.stack([
        np.array(train_df[:w2.total_window_size]),
        np.array(train_df[100:100+w2.total_window_size]),
        np.array(train_df[200:200+w2.total_window_size])
    ])
    example_inputs, example_labels = w2.split_window(example_window)

    if args.step == 7:
        print('All shapes are: (batch, time, features)')
        print(f'Window shape: {example_window.shape}')
        print(f'Inputs shape: {example_inputs.shape}')
        print(f'labels shape: {example_labels.shape}')

        if args.plot:
            plt.figure()
            img = tf.io.read_file('supplement/tf2_t0904_03.png')
            img = tf.image.decode_png(img)
            plt.imshow(img)
            plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #8 - Data windowing: Plot
if args.step >= 8: 
    print("\n### Step #8 - Data windowing: Plot")

    w2.example = example_inputs, example_labels
   
    def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(
                self.input_indices, inputs[n, :, plot_col_index],
                label='Inputs', marker='.', zorder=-10
            )

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(
                self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64
            )
            if model is not None:
                predictions = model(inputs)
                plt.scatter(
                    self.label_indices, predictions[n, :, label_col_index],
                    marker='X', edgecolors='k', label='Predictions',
                    c='#ff7f0e', s=64
                )

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
        plt.show(block=False)

    WindowGenerator.plot = plot

    if args.step == 8 and args.plot:
        w2.plot()
        w2.plot(plot_col='p (mbar)')


args.step = auto_increment(args.step, args.all)
### Step #9 - Data windowing: Create tf.data.Datasets
if args.step >= 9: 
    print("\n### Step #9 - Data windowing: Create tf.data.Datasets")

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=32,
        )

        ds = ds.map(self.split_window)
        return ds

    WindowGenerator.make_dataset = make_dataset

    @property
    def train(self):
      return self.make_dataset(self.train_df)

    @property
    def val(self):
      return self.make_dataset(self.val_df)

    @property
    def test(self):
      return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    WindowGenerator.train = train
    WindowGenerator.val = val
    WindowGenerator.test = test
    WindowGenerator.example = example

    if args.step == 9:
        # Each element is an (inputs, label) pair
        logger.info('w2.train.element_spec:')
        print(*w2.train.element_spec, sep='\n')
        print()

        logger.info('Inputs and Labels shape:')
        for example_inputs, example_labels in w2.train.take(1):
            print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
            print(f'Labels shape (batch, time, features): {example_labels.shape}')


args.step = auto_increment(args.step, args.all)
### Step #10 - Single step models
if args.step >= 10: 
    print("\n### Step #10 - Single step models")

    single_step_window = WindowGenerator(
        input_width=1, label_width=1, shift=1, label_columns=['T (degC)']
    )
    if args.step == 10:
        print(single_step_window, '\n')

        for example_inputs, example_labels in single_step_window.train.take(1):
            print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
            print(f'Labels shape (batch, time, features): {example_labels.shape}')


args.step = auto_increment(args.step, args.all)
### Step #11 - Single step models: Baseline
if args.step >= 11: 
    print("\n### Step #11 - Single step models: Baseline")

    class Baseline(tf.keras.Model):
        def __init__(self, label_index=None):
            super().__init__()
            self.label_index = label_index

        def call(self, inputs):
            if self.label_index is None:
                return inputs
            result = inputs[:, :, self.label_index]
            return result[:, :, tf.newaxis]

    baseline = Baseline(label_index=column_indices['T (degC)'])

    baseline.compile(
        loss=tf.losses.MeanSquaredError(),
        metrics=[tf.metrics.MeanAbsoluteError()]
    )

    val_performance = {}
    performance = {}
    val_performance['Baseline'] = baseline.evaluate(single_step_window.val, verbose=0)
    performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)

    wide_window = WindowGenerator(
        input_width=24, label_width=24, shift=1, label_columns=['T (degC)']
    )
    if args.step == 11 and args.plot:
        logger.info(f'wide_window:\n{wide_window}\n')
        logger.info(f'Input shape: {wide_window.example[0].shape}')
        logger.info(f'Output shape: {baseline(wide_window.example[0]).shape}')
        wide_window.plot(baseline)


args.step = auto_increment(args.step, args.all)
### Step #12 - Single step models: Linear model
if args.step >= 12: 
    print("\n### Step #12 - Single step models: Linear model")

    linear = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1)
    ])

    MAX_EPOCHS = args.epochs # 20

    def compile_and_fit(model, window, patience=2, verbose=0):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, mode='min'
        )

        model.compile(
            loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(),
            metrics=[tf.metrics.MeanAbsoluteError()]
        )

        history = model.fit(
            window.train, 
            epochs=MAX_EPOCHS,
            validation_data=window.val,
            callbacks=[early_stopping],
            verbose=verbose
        )

        return history

    if args.step == 12:
        logger.info(f'Input shape: {single_step_window.example[0].shape}')
        logger.info(f'Output shape: {linear(single_step_window.example[0]).shape}\n')

    history = compile_and_fit(linear, single_step_window, verbose=args.step==12)

    val_performance['Linear'] = linear.evaluate(single_step_window.val, verbose=0)
    performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)

    if args.step == 12 and args.plot:
        logger.info(f'Input shape: {wide_window.example[0].shape}')
        logger.info(f'Output shape: {linear(wide_window.example[0]).shape}')
        wide_window.plot(linear)

        plt.figure()
        plt.bar(x = range(len(train_df.columns)),
        height=linear.layers[0].kernel[:,0].numpy())
        axis = plt.gca()
        axis.set_xticks(range(len(train_df.columns)))
        _ = axis.set_xticklabels(train_df.columns, rotation=90)
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #13 - Single step models: Dense
if args.step >= 13: 
    print("\n### Step #13 - Single step models: Dense")

    dense = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])

    history = compile_and_fit(dense, single_step_window, verbose=args.step==13)

    val_performance['Dense'] = dense.evaluate(single_step_window.val, verbose=0)
    performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)

    if args.step == 13 and args.plot:
        wide_window.plot(dense)


args.step = auto_increment(args.step, args.all)
### Step #14 - Single step models: Multi-step dense
if args.step >= 14: 
    print("\n### Step #14 - Single step models: Multi-step dense")

    CONV_WIDTH = 3
    conv_window = WindowGenerator(
        input_width=CONV_WIDTH,
        label_width=1,
        shift=1,
        label_columns=['T (degC)'])

    if args.step == 14:
        print(conv_window, '\n')
        if args.plot:
            conv_window.plot()

    multi_step_dense = tf.keras.Sequential([
        # Shape: (time, features) => (time*features)
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
        # Add back the time dimension.
        # Shape: (outputs) => (1, outputs)
        tf.keras.layers.Reshape([1, -1]),
    ])

    if args.step == 14:
        logger.info(f'Input shape: {conv_window.example[0].shape}')
        logger.info(f'Output shape: {multi_step_dense(conv_window.example[0]).shape}')

    history = compile_and_fit(multi_step_dense, conv_window, verbose=args.step==14)

    val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val, verbose=0)
    performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0)

    if args.step == 14:
        logger.info(f'conv_window:\n{conv_window}\n')
        logger.info(f'Input shape: {conv_window.example[0].shape}')
        logger.info(f'Output shape: {multi_step_dense(conv_window.example[0]).shape}')

        if args.plot:
            conv_window.plot()
            conv_window.plot(multi_step_dense)


args.step = auto_increment(args.step, args.all)
### Step #15 - Single step models: Convolution neural network
if args.step >= 15: 
    print("\n### Step #15 - Single step models: Convolution neural network")

    conv_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(
            filters=32, kernel_size=(CONV_WIDTH,), activation='relu'
        ),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
    ])

    if args.step == 15:
        logger.info("Conv model on conv_window:")
        print(f'Input shape: {conv_window.example[0].shape}')
        print(f'Output shape: {conv_model(conv_window.example[0]).shape}')


    history = compile_and_fit(conv_model, conv_window, verbose=args.step==15)
    val_performance['Conv'] = conv_model.evaluate(conv_window.val, verbose=0)
    performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)

    LABEL_WIDTH = 24
    INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
    wide_conv_window = WindowGenerator(
        input_width=INPUT_WIDTH,
        label_width=LABEL_WIDTH,
        shift=1,
        label_columns=['T (degC)']
    )

    if args.step == 15:
        logger.info(f'wide_conv_window:\n{wide_conv_window}\n')
        logger.info(f'Input shape: {wide_conv_window.example[0].shape}')
        logger.info(f'Labels shape: {wide_conv_window.example[1].shape}')
        logger.info(f'Output shape: {conv_model(wide_conv_window.example[0]).shape}')

        if args.plot:
            wide_conv_window.plot(conv_model)


args.step = auto_increment(args.step, args.all)
### Step #16 - Single step models: Recurrent neural network
if args.step >= 16: 
    print("\n### Step #16 - Single step models: Recurrent neural network")

    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])

    history = compile_and_fit(lstm_model, wide_window, verbose=args.step==16)
    val_performance['LSTM'] = lstm_model.evaluate(wide_window.val, verbose=0)
    performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)

    if args.step == 16:
        logger.info("lstm model on wide_window:")
        print('Input shape:', wide_window.example[0].shape)
        print('Output shape:', lstm_model(wide_window.example[0]).shape)

        if args.plot:
            wide_window.plot(lstm_model)


args.step = auto_increment(args.step, args.all)
### Step #17 - Single step models: Performance
if args.step == 17: 
    print("\n### Step #17 - Single step models: Performance")

    for name, value in performance.items():
        print(f'{name:12s}: {value[1]:0.4f}')

    if args.plot:
        x = np.arange(len(performance))
        width = 0.3
        metric_name = 'mean_absolute_error'
        metric_index = lstm_model.metrics_names.index('mean_absolute_error')
        val_mae = [v[metric_index] for v in val_performance.values()]
        test_mae = [v[metric_index] for v in performance.values()]

        plt.figure()
        plt.ylabel('mean_absolute_error [T (degC), normalized]')
        plt.bar(x - 0.17, val_mae, width, label='Validation')
        plt.bar(x + 0.17, test_mae, width, label='Test')
        plt.xticks(ticks=x, labels=performance.keys(), rotation=45)
        _ = plt.legend()
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #18 - Single step models: Multi-output models
if args.step == 18: 
    print("\n### Step #18 - Single step models: Multi-output models")


args.step = auto_increment(args.step, args.all)
### Step #19 - Multi-step models
if args.step >= 19: 
    print("\n### Step #19 - Multi-step models")


args.step = auto_increment(args.step, args.all)
### Step #19 - Multi-step models: Baselines
if args.step >= 19: 
    print("\n### Step #19 - Multi-step models: Baselines")


args.step = auto_increment(args.step, args.all)
### Step #20 - Multi-step models: Single-shot models
if args.step >= 20: 
    print("\n### Step #20 - Multi-step models: Single-shot models")


args.step = auto_increment(args.step, args.all)
### Step #21 - Multi-step models: Autoregressive model
if args.step >= 21: 
    print("\n### Step #21 - Multi-step models: Autoregressive model")


args.step = auto_increment(args.step, args.all)
### Step #22 - Multi-step models: Performance
if args.step >= 22: 
    print("\n### Step #22 - Multi-step models: Performance")


args.step = auto_increment(args.step, args.all)
### Step #23 - Next steps
if args.step >= 23: 
    print("\n### Step #23 - Next steps")



### End of File
print()
if args.plot:
    plt.show()
debug()


