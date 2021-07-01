#!/usr/bin/env python

import sys
sys.path.append('./')
sys.path.append('../../')

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
import io
import re
import shutil
import string

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - Representing text as numbers
if args.step == 1: 
    print("\n### Step #1 - Representing text as numbers")

    logger.info('One-hot encodings:')
    __doc__ ='''
    This approach is inefficient.
    A one-hot encoded vector is sparse (meaning, most indices are zero).
    Imagine you have 10,000 words in the vocabulary. To one-hot encode each word,
    you would create a vector where 99.99% of the elements are zero.
    '''
    print(__doc__)

    logger.info('Encode each word with a unique number:')
    __doc__='''
    This appoach is efficient. Instead of a sparse vector,
    you now have a dense one (where all elements are full).
    There are two downsides to this approach, however:
    - The integer-encoding is arbitrary
      (it does not capture any relationship between words).
    - An integer-encoding can be challenging for a model to interpret.
      A linear classifier, for example, learns a single weight for each
      feature. Because there is no relationship between the similarity
      of any two words and the similarity of their encodings, this feature-weight
      combination is not meaningful.
    '''
    print(__doc__)

    logger.info('Word embeddings:')
    __doc__='''
    Word embeddings give us a way to use an efficient, dense representation
    in which similar words have a similar encoding. Importantly, you do not
    have to specify this encoding by hand.  An embedding is a dense vector of
    floating point values (the length of the vector is a parameter you specify).
    Instead of specifying the values for the embedding manually, they are
    trainable parameters (weights learned by the model during training).
    It is common to see word embeddings that are 8-dimensional (for small
    datasets), up to 1024-dimensions when working with large datasets.
    A higher dimensional embedding can capture fine-grained relationships
    between words, but takes more data to learn.
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #2 - Setup: Download the IMDb Dataset
if args.step in [2, 3, 5, 6, 7, 8]: 
    print("\n### Step #2 - Setup: Download the IMDb Dataset")

    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    dataset = tf.keras.utils.get_file(
        "aclImdb_v1.tar.gz", 
        url,
        untar=True
    )

    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
    train_dir = os.path.join(dataset_dir, 'train')
    remove_dir = os.path.join(train_dir, 'unsup')
    if os.path.exists(remove_dir):
        shutil.rmtree(remove_dir)
    
    batch_size = 1024
    seed = 123

    train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        train_dir, batch_size=batch_size, validation_split=0.2,
        subset='training', seed=seed
    )
    val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        train_dir, batch_size=batch_size, validation_split=0.2,
        subset='validation', seed=seed
    )

    if args.step == 2:
        print()
        logger.info(dataset_dir)
        print(*os.listdir(dataset_dir), sep='\n')
        print()
        logger.info('samples:')
        for text_batch, label_batch in train_ds.take(1):
            for i in range(5):
                print(label_batch[i].numpy(), ':', text_batch.numpy()[i][:50])


args.step = auto_increment(args.step, args.all)
### Step #3 - Setup: Configure the dataset for performance
if args.step in [3, 5, 6, 7, 8]: 
    print("\n### Step #3 - Setup: Configure the dataset for performance")

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


args.step = auto_increment(args.step, args.all)
### Step #4 - Using the Embedding layer
if args.step == 4: 
    print("\n### Step #4 - Using the Embedding layer")

    # Embed a 1,000 word vocabulary into 5 dimensions.
    embedding_layer = tf.keras.layers.Embedding(1000, 5)

    result = embedding_layer(tf.constant([1, 2, 3]))
    logger.info('embedding_layer(tf.constant([1, 2, 3])):')
    print(result.numpy(), '\n')

    input_text = tf.constant([[0, 1, 2], [3, 4, 5]])
    result = embedding_layer(input_text)
    logger.info(f'shape of input_text: {input_text.shape}')
    logger.info(f'shape of embedding_layer: {result.shape}')


args.step = auto_increment(args.step, args.all)
### Step #5 - Text preprocessing
if args.step in [5, 6, 7, 8]: 
    print("\n### Step #5 - Text preprocessing")

    # Create a custom standardization function to strip HTML break tags '<br />'.
    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        return tf.strings.regex_replace(
            stripped_html, '[%s]' % re.escape(string.punctuation), ''
        )

    # Vocabulary size and number of words in a sequence.
    vocab_size = 10000
    sequence_length = 100

    # Use the text vectorization layer to normalize, split, and map strings to
    # integers. Note that the layer uses the custom standardization defined above.
    # Set maximum_sequence length as all samples are not of the same length.
    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length
    )

    # Make a text-only dataset (no labels) and call adapt to build the vocabulary.
    text_ds = train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(text_ds)


args.step = auto_increment(args.step, args.all)
### Step #6 - Create a classification model
if args.step in [6, 7, 8]: 
    print("\n### Step #6 - Create a classification model")

    embedding_dim=16
    model = Sequential([
        vectorize_layer,
        Embedding(vocab_size, embedding_dim, name="embedding"),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1)
    ])


args.step = auto_increment(args.step, args.all)
### Step #7 - Compile and train the model
if args.step in [7, 8]: 
    print("\n### Step #7 - Compile and train the model")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="tmp/tf2_t0701/logs")

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs, # 15
        callbacks=[tensorboard_callback],
        verbose=2 if args.step == 7 else 0
    )

    if args.step == 7:
        print()
        model.summary()
        print('\ntensorboard --logdir tmp/tf2_t0701/logs --bind_all')


args.step = auto_increment(args.step, args.all)
### Step #8 - Retrieve the trained word embeddings and save them to disk
if args.step == 8: 
    print("\n### Step #8 - Retrieve the trained word embeddings and save them to disk")

    weights = model.get_layer('embedding').get_weights()[0]
    vocab = vectorize_layer.get_vocabulary()

    out_v = io.open('tmp/tf2_t0701/vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open('tmp/tf2_t0701/metadata.tsv', 'w', encoding='utf-8')

    for index, word in enumerate(vocab):
        if index == 0:
            continue  # skip 0, it's padding.
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()


args.step = auto_increment(args.step, args.all)
### Step #9 - Visualize the embeddings
if args.step == 9:
    print("\n### Step #9 - Visualize the embeddings")

    __doc__ = '''
    To visualize the embeddings, upload them to the embedding projector.
    1. Open the Embedding Projector[http://projector.tensorflow.org/]
       (this can also run in a local TensorBoard instance).
    2. Click on "Load data".
    3. Upload the two files you created above: vecs.tsv and meta.tsv.
    '''
    print(__doc__)



### End of File
print()
if args.plot:
    plt.show()
debug()


