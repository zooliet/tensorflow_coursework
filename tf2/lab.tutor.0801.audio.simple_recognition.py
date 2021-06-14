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
import pathlib
import seaborn as sns

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("\n#################################################")
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Softmax, InputLayer
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers.experimental import preprocessing


### TOC
if args.step == 0:
    toc(__file__)
args.step = auto_increment(args.step, args.all)


if True:
    # Set seed for experiment reproducibility
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)


### Step #1 - Import the Speech Commands dataset
if args.step >= 1: 
    print("\n### Step #1 - Import the Speech Commands dataset")

    data_dir = pathlib.Path(f'{os.getenv("HOME")}/.keras/datasets/mini_speech_commands')
    if not data_dir.exists():
        tf.keras.utils.get_file(
            'mini_speech_commands.zip',
            origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
            extract=True,
            # cache_subdir='datasets/speech_data'
        )

    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    commands = commands[commands != 'README.md']
    if args.step == 1:
        logger.info(f'Commands: {commands}')

    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
    filenames = tf.random.shuffle(filenames)
    num_samples = len(filenames)
    if args.step == 1:
        logger.info(f'Number of total examples: {num_samples}')
        logger.info(f'Number of examples per label: {len(tf.io.gfile.listdir(str(data_dir/commands[0])))}')
        logger.info(f'Example file tensor:\n{filenames[0]}')

    # Split the files into training, validation and test sets using a 80:10:10 ratio, respectively
    train_files = filenames[:6400]
    val_files = filenames[6400: 6400 + 800]
    test_files = filenames[-800:]

    if args.step == 1:
        logger.info(f'Training set size: {len(train_files)}')
        logger.info(f'Validation set size: {len(val_files)}')
        logger.info(f'Test set size: {len(test_files)}')


args.step = auto_increment(args.step, args.all)
### Step #2 - Reading audio files and their labels
if args.step >= 2: 
    print("\n### Step #2 - Reading audio files and their labels")

    def decode_audio(audio_binary):
        audio, _ = tf.audio.decode_wav(audio_binary)
        return tf.squeeze(audio, axis=-1)

    def get_label(file_path):
        parts = tf.strings.split(file_path, os.path.sep)

        # Note: You'll use indexing here instead of tuple unpacking to enable this
        # to work in a TensorFlow graph.
        return parts[-2]

    def get_waveform_and_label(file_path):
        label = get_label(file_path)
        audio_binary = tf.io.read_file(file_path)
        waveform = decode_audio(audio_binary)
        return waveform, label

    AUTOTUNE = tf.data.AUTOTUNE
    files_ds = tf.data.Dataset.from_tensor_slices(train_files)
    waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

    if args.step == 2 and args.plot:
        rows = 3
        cols = 3
        n = rows*cols
        fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
        for i, (audio, label) in enumerate(waveform_ds.take(n)):
            r = i // cols
            c = i % cols
            ax = axes[r][c]
            ax.plot(audio.numpy())
            ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
            label = label.numpy().decode('utf-8')
            ax.set_title(label)
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #3 - Spectrogram
if args.step >= 3: 
    print("\n### Step #3 - Spectrogram")

    def get_spectrogram(waveform):
        # Padding for files with less than 16000 samples
        zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

        # Concatenate audio with padding so that all audio clips will be of the
        # same length
        waveform = tf.cast(waveform, tf.float32)
        equal_length = tf.concat([waveform, zero_padding], 0)
        spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)

        return spectrogram

    def plot_spectrogram(spectrogram, ax):
        # Convert to frequencies to log scale and transpose so that the time is
        # represented in the x-axis (columns).
        log_spec = np.log(spectrogram.T)
        height = log_spec.shape[0]
        width = log_spec.shape[1]
        X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
        Y = range(height)
        ax.pcolormesh(X, Y, log_spec)

    if args.step == 3:
        for waveform, label in waveform_ds.take(1):
            label = label.numpy().decode('utf-8')
            spectrogram = get_spectrogram(waveform)

        logger.info(f'Label: {label}')
        logger.info(f'Waveform shape: {waveform.shape}')
        logger.info(f'Spectrogram shape: {spectrogram.shape}')
        # logger.info(f'Audio playback')
        # display.display(display.Audio(waveform, rate=16000))

        if args.plot:
            fig, axes = plt.subplots(2, figsize=(12, 8))
            timescale = np.arange(waveform.shape[0])
            axes[0].plot(timescale, waveform.numpy())
            axes[0].set_title('Waveform')
            axes[0].set_xlim([0, 16000])
            plot_spectrogram(spectrogram.numpy(), axes[1])
            axes[1].set_title('Spectrogram')
            plt.show(block=False)

    def get_spectrogram_and_label_id(audio, label):
        spectrogram = get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        label_id = tf.argmax(label == commands)
        return spectrogram, label_id

    spectrogram_ds = waveform_ds.map(
        get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE
    )

    if args.step == 3 and args.plot:
        rows = 3
        cols = 3
        n = rows*cols
        fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
        for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
            r = i // cols
            c = i % cols
            ax = axes[r][c]
            plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
            ax.set_title(commands[label_id.numpy()])
            ax.axis('off')

        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #4 - Build and train the model
if args.step >= 4: 
    print("\n### Step #4 - Build and train the model")

    def preprocess_dataset(files):
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
        output_ds = output_ds.map(
            get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE
        )
        return output_ds

    train_ds = spectrogram_ds
    val_ds = preprocess_dataset(val_files)
    test_ds = preprocess_dataset(test_files)

    batch_size = args.batch # 64
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    for spectrogram, _ in spectrogram_ds.take(1):
        input_shape = spectrogram.shape # (124, 129, 1)

    # logger.info(f'Input shape: {input_shape}')
    num_labels = len(commands)

    norm_layer = preprocessing.Normalization()
    norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

    model = Sequential([
        InputLayer(input_shape=input_shape),
        preprocessing.Resizing(32, 32),
        norm_layer,
        Conv2D(32, 3, activation='relu'),
        Conv2D(64, 3, activation='relu'),
        MaxPooling2D(),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_labels),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
        verbose=2 if args.step == 4 else 0
    )

    if args.step == 4:
        model.summary()

        if args.plot:
            metrics = history.history
            plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
            plt.legend(['loss', 'val_loss'])
            plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #5 - Evaluate test set performance
if args.step in [5, 6]: 
    print("\n### Step #5 - Evaluate test set performance")

    test_audio = []
    test_labels = []

    for audio, label in test_ds:
        test_audio.append(audio.numpy())
        test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels

    test_acc = sum(y_pred == y_true) / len(y_true)
    if args.step == 5:
        logger.info(f'Test set accuracy: {test_acc:.0%}')


args.step = auto_increment(args.step, args.all)
### Step #6 - Evaluate test set performance: Display a confusion matrix
if args.step == 6: 
    print("\n### Step #6 - Evaluate test set performance: Display a confusion matrix")

    if args.plot:
        confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion_mtx, 
            xticklabels=commands, yticklabels=commands,
            annot=True, fmt='g'
        )
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #7 - Run inference on an audio file
if args.step == 7: 
    print("\n### Step #7 - Run inference on an audio file")

    sample_file = data_dir/'no/01bb6a2a_nohash_0.wav'
    sample_ds = preprocess_dataset([str(sample_file)])

    for spectrogram, label in sample_ds.batch(1):
        prediction = model(spectrogram)
        logger.info(f'prediction:\n{prediction}')
        if args.plot:
            plt.bar(commands, tf.nn.softmax(prediction[0]))
            plt.title(f'Predictions for "{commands[label[0]]}"')
            plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #8 - Next steps
if args.step == 8: 
    print("\n### Step #8 - Next steps")



### End of File
if args.plot:
    plt.show()
debug()


