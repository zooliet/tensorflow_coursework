#!/usr/bin/env python

# pip install -q tensorflow_io 

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
from PIL import Image

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import InputLayer, Layer, Dense 

import tensorflow_hub as hub
import tensorflow_io as tfio


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - About YAMNet: Loading YAMNet from TensorFlow Hub
if args.step >= 1: 
    print("\n### Step #1 - About YAMNet: Loading YAMNet from TensorFlow Hub")

    yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
    yamnet_model = hub.load(yamnet_model_handle)

    testing_wav_file_name = tf.keras.utils.get_file(
        'miaow_16k.wav',
        'https://storage.googleapis.com/audioset/miaow_16k.wav',
    )

    # Util functions for loading audio files and ensure the correct sample rate
    @tf.function
    def load_wav_16k_mono(filename):
        """ read in a waveform file and convert to 16 kHz mono """
        file_contents = tf.io.read_file(filename)
        wav, sample_rate = tf.audio.decode_wav(
              file_contents,
              desired_channels=1
        )
        wav = tf.squeeze(wav, axis=-1)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
        return wav

    testing_wav_data = load_wav_16k_mono(testing_wav_file_name)

    if args.step == 1:
        logger.info(f'sample file: {testing_wav_file_name}')

        if args.plot:
            _ = plt.plot(testing_wav_data)
            plt.show(block=False)
            # # Play the audio file.
            # display.Audio(testing_wav_data,rate=16000)


args.step = auto_increment(args.step, args.all)
### Step #2 - About YAMNet: Load the class mapping
if args.step >= 2: 
    print("\n### Step #2 - About YAMNet: Load the class mapping")

    class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
    class_names =list(pd.read_csv(class_map_path)['display_name'])

    if args.step == 2:
        logger.info(f'yamnet_model class_map_path: {class_map_path}')
        logger.info('class_names:')
        for name in class_names[:20]:
          print(name)
        print('...')


args.step = auto_increment(args.step, args.all)
### Step #3 - About YAMNet: Run inference
if args.step == 3: 
    print("\n### Step #3 - About YAMNet: Run inference")

    scores, embeddings, spectrogram = yamnet_model(testing_wav_data)
    class_scores = tf.reduce_mean(scores, axis=0)
    top_class = tf.argmax(class_scores)
    infered_class = class_names[top_class]

    if args.step == 3:
        logger.info(f'The main sound is: {infered_class}')
        logger.info(f'The embeddings shape: {embeddings.shape}')


args.step = auto_increment(args.step, args.all)
### Step #4 - ESC-50 dataset
if args.step >= 4: 
    print("\n### Step #4 - ESC-50 dataset")

    zipped_dataset_path = tf.keras.utils.get_file(
        'esc-50.zip',
        'https://github.com/karoldvl/ESC-50/archive/master.zip',
        extract=True
    )


args.step = auto_increment(args.step, args.all)
### Step #5 - ESC-50 dataset: Explore the data
if args.step >= 5: 
    print("\n### Step #5 - ESC-50 dataset: Explore the data")

    dataset_dir = os.path.dirname(zipped_dataset_path) 
    esc50_csv = f'{dataset_dir}/ESC-50-master/meta/esc50.csv'
    base_data_path = f'{dataset_dir}/ESC-50-master/audio/'

    pd_data = pd.read_csv(esc50_csv)
    if args.step == 5:
        logger.info('esc50.csv:')
        print(pd_data.head())


args.step = auto_increment(args.step, args.all)
### Step #6 - ESC-50 dataset: Filter the data
if args.step >= 6: 
    print("\n### Step #6 - ESC-50 dataset: Filter the data")

    my_classes = ['dog', 'cat']
    map_class_to_id = {'dog':0, 'cat':1}

    filtered_pd = pd_data[pd_data.category.isin(my_classes)]

    class_id = filtered_pd['category'].apply(lambda name: map_class_to_id[name])
    filtered_pd = filtered_pd.assign(target=class_id)

    full_path = filtered_pd['filename'].apply(lambda row: os.path.join(base_data_path, row))
    filtered_pd = filtered_pd.assign(filename=full_path)

    if args.step == 6:
        logger.info('After filtered:')
        print(filtered_pd.head(10))


args.step = auto_increment(args.step, args.all)
### Step #7 - ESC-50 dataset: Load the audio files and retrieve embeddings
if args.step >= 7: 
    print("\n### Step #7 - ESC-50 dataset: Load the audio files and retrieve embeddings")

    filenames = filtered_pd['filename']
    targets = filtered_pd['target']
    folds = filtered_pd['fold']

    main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))
    if args.step == 7:
        print(*list(main_ds.element_spec), sep='\n')
    print('')
    
    def load_wav_for_map(filename, label, fold):
        return load_wav_16k_mono(filename), label, fold

    main_ds = main_ds.map(load_wav_for_map)
    if args.step == 7:
        print(*list(main_ds.element_spec), sep='\n')
    print('')

    # applies the embedding extraction model to a wav data
    def extract_embedding(wav_data, label, fold):
        ''' run YAMNet to extract embedding from the wav data '''
        scores, embeddings, spectrogram = yamnet_model(wav_data)
        num_embeddings = tf.shape(embeddings)[0]
        return (
            embeddings,
            tf.repeat(label, num_embeddings),
            tf.repeat(fold, num_embeddings)
        )

    # extract embedding
    main_ds = main_ds.map(extract_embedding).unbatch()
    if args.step == 7:
        print(*list(main_ds.element_spec), sep='\n')
    print('')


args.step = auto_increment(args.step, args.all)
### Step #8 - ESC-50 dataset: Split the data
if args.step >= 8: 
    print("\n### Step #8 - ESC-50 dataset: Split the data")

    cached_ds = main_ds.cache()
    train_ds = cached_ds.filter(lambda embedding, label, fold: fold < 4)
    val_ds = cached_ds.filter(lambda embedding, label, fold: fold == 4)
    test_ds = cached_ds.filter(lambda embedding, label, fold: fold == 5)

    # remove the folds column now that it's not needed anymore
    remove_fold_column = lambda embedding, label, fold: (embedding, label)

    train_ds = train_ds.map(remove_fold_column)
    val_ds = val_ds.map(remove_fold_column)
    test_ds = test_ds.map(remove_fold_column)

    train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)


args.step = auto_increment(args.step, args.all)
### Step #9 - Create your model
if args.step >= 9: 
    print("\n### Step #9 - Create your model")

    my_model = Sequential([
        InputLayer(input_shape=(1024), dtype=tf.float32, name='input_embedding'),
        Dense(512, activation='relu'),
        Dense(len(my_classes))
    ], name='my_model')

    my_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer="adam",
        metrics=['accuracy']
    )

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=3,
        restore_best_weights=True
    )

    history = my_model.fit(
        train_ds,
        epochs=args.epochs, # 20,
        validation_data=val_ds,
        callbacks=callback,
        verbose=2 if args.step == 9 else 0
    )

    loss, accuracy = my_model.evaluate(test_ds, verbose=0)

    if args.step == 9:
        logger.info(f"Loss: {loss:.2f}")
        logger.info(f"Accuracy: {accuracy:.2f}")

        my_model.summary()


args.step = auto_increment(args.step, args.all)
### Step #10 - Test your model
if args.step == 10: 
    print("\n### Step #10 - Test your model")

    scores, embeddings, spectrogram = yamnet_model(testing_wav_data)
    result = my_model(embeddings).numpy()

    infered_class = my_classes[result.mean(axis=0).argmax()]
    logger.info(f'The main sound is: {infered_class}')


args.step = auto_increment(args.step, args.all)
### Step #11 - Save a model that can directly take a wav file as input
if args.step >= 11: 
    print("\n### Step #11 - Save a model that can directly take a wav file as input")

    class ReduceMeanLayer(Layer):
        def __init__(self, axis=0, **kwargs):
            super(ReduceMeanLayer, self).__init__(**kwargs)
            self.axis = axis

        def call(self, input):
            return tf.math.reduce_mean(input, axis=self.axis)

    saved_model_path = 'tmp/dogs_and_cats_yamnet'

    input_segment = Input(shape=(), dtype=tf.float32, name='audio')
    embedding_extraction_layer = hub.KerasLayer(
        yamnet_model_handle,
        trainable=False, name='yamnet'
    )
    _, embeddings_output, _ = embedding_extraction_layer(input_segment)
    serving_outputs = my_model(embeddings_output)
    serving_outputs = ReduceMeanLayer(axis=0, name='classifier')(serving_outputs)
    serving_model = Model(input_segment, serving_outputs)
    serving_model.save(saved_model_path, include_optimizer=False)

    tf.keras.utils.plot_model(serving_model, 'tmp/dogs_and_cats_yamnet.png')
    if args.step == 11 and args.plot:
        image = Image.open('tmp/segmentation.png')
        plt.figure()
        plt.imshow(image)
        plt.show(block=False)
    

    reloaded_model = tf.saved_model.load(saved_model_path)
    reloaded_results = reloaded_model(testing_wav_data)
    cat_or_dog = my_classes[tf.argmax(reloaded_results)]
    if args.step == 11:
        logger.info(f'The main sound is: {cat_or_dog}')

    # If you want to try your new model on a serving setup, you can use the 'serving_default' signature
    serving_results = reloaded_model.signatures['serving_default'](testing_wav_data)
    cat_or_dog = my_classes[tf.argmax(serving_results['classifier'])]
    if args.step == 11:
        logger.info(f'The main sound is: {cat_or_dog}')


args.step = auto_increment(args.step, args.all)
### Step #12 - (Optional) Some more testing
if args.step == 12: 
    print("\n### Step #12 - (Optional) Some more testing")

    test_pd = filtered_pd.loc[filtered_pd['fold'] == 5]
    row = test_pd.sample(1)
    filename = row['filename'].item()
    logger.info(f'filename: {filename}')
    
    waveform = load_wav_16k_mono(filename)
    logger.info(f'Waveform values: {waveform}')

    if args.plot:
        _ = plt.plot(waveform)
        # display.Audio(waveform, rate=16000)

    # Run the model, check the output.
    scores, embeddings, spectrogram = yamnet_model(waveform)
    class_scores = tf.reduce_mean(scores, axis=0)
    top_class = tf.argmax(class_scores)
    infered_class = class_names[top_class]
    top_score = class_scores[top_class]
    logger.info(f'[YAMNet] The main sound is: {infered_class} ({top_score})')

    reloaded_results = reloaded_model(waveform)
    your_top_class = tf.argmax(reloaded_results)
    your_infered_class = my_classes[your_top_class]
    class_probabilities = tf.nn.softmax(reloaded_results, axis=-1)
    your_top_score = class_probabilities[your_top_class]
    logger.info(f'[Your model] The main sound is: {your_infered_class} ({your_top_score})')

### End of File
if args.plot:
    plt.show()
debug()


