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
import re
import string

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Dot, Embedding, Flatten
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import tqdm


### TOC
if args.step == 0:
    toc(__file__)


SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

args.step = auto_increment(args.step, args.all)
### Step #1 - Setup: Vectorize an example sentence
if args.step in [1, 2, 3, 4]: 
    print("\n### Step #1 - Setup: Vectorize an example sentence")

    sentence = "The wide road shimmered in the hot sun"
    tokens = list(sentence.lower().split())

    # Create a vocabulary to save mappings from tokens to integer indices
    vocab, index = {}, 1  # start indexing from 1
    vocab['<pad>'] = 0  # add a padding token
    for token in tokens:
        if token not in vocab:
            vocab[token] = index
            index += 1
    vocab_size = len(vocab)

    # Create an inverse vocabulary to save mappings from integer indices to tokens.
    inverse_vocab = {index: token for token, index in vocab.items()}

    if args.step == 1:
        logger.info(f'vocab size: {vocab_size}')
        logger.info('vocab:')
        print(*list(vocab.items()), sep='\n')
        logger.info('inverse_vocab:')
        print(*list(inverse_vocab.items()), sep='\n')

    # vectorize your sentence
    example_sequence = [vocab[word] for word in tokens]

    if args.step == 1:
        logger.info(f'{sentence} => ')
        logger.info(f'{example_sequence}\n')


args.step = auto_increment(args.step, args.all)
### Step #2 - Setup: Generate skip-grams from one sentence
if args.step in [2, 3, 4]: 
    print("\n### Step #2 - Setup: Generate skip-grams from one sentence")

    window_size = 2
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
        example_sequence,
        vocabulary_size=vocab_size,
        window_size=window_size,
        negative_samples=0
    )

    if args.step == 2:
        logger.info(f'# of positive_skip_grams: {len(positive_skip_grams)}')
        logger.info('Take a look at few positive skip-grams:')

        for target, context in positive_skip_grams[:5]:
            print(f"({target}, {context}): ({inverse_vocab[target]}, {inverse_vocab[context]})")
        print('')


args.step = auto_increment(args.step, args.all)
### Step #3 - Setup: Negative sampling for one skip-gram
if args.step in [3, 4]: 
    print("\n### Step #3 - Setup: Negative sampling for one skip-gram")

    # Get target and context words for one positive skip-gram.
    target_word, context_word = positive_skip_grams[0]

    # Set the number of negative samples per positive context.
    num_ns = 4

    context_class = tf.reshape(tf.constant(context_word, dtype="int64"), (1, 1))
    negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
        true_classes=context_class,  # class that should be sampled as 'positive'
        num_true=1,  # each positive skip-gram has 1 positive context class
        num_sampled=num_ns,  # number of negative context words to sample
        unique=True,  # all the negative samples should be unique
        range_max=vocab_size,  # pick index of the samples from [0, vocab_size]
        seed=SEED,  # seed for reproducibility
        name="negative_sampling"  # name of this operation
    )

    if args.step == 3:
        logger.info('negative_sampling_candidates:')
        print([inverse_vocab[index.numpy()] for index in negative_sampling_candidates])


args.step = auto_increment(args.step, args.all)
### Step #4 - Setup: Construct one training example
if args.step == 4: 
    print("\n### Step #4 - Setup: Construct one training example")

    # Add a dimension so you can use concatenation (on the next step).
    negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)

    # Concat positive context word with negative sampled words.
    context = tf.concat([context_class, negative_sampling_candidates], 0)

    # Label first context word as 1 (positive) followed by num_ns 0s (negative).
    label = tf.constant([1] + [0]*num_ns, dtype="int64")

    # Reshape target to shape (1,) and context and label to (num_ns+1,).
    target = tf.squeeze(target_word)
    context = tf.squeeze(context)
    label = tf.squeeze(label)

    # Take a look at the context and the corresponding labels for the target word 
    # from the skip-gram example above.

    print(f"target_index    : {target}")
    print(f"target_word     : {inverse_vocab[target_word]}")
    print(f"context_indices : {context}")
    print(f"context_words   : {[inverse_vocab[c.numpy()] for c in context]}")
    print(f"label           : {label}")


args.step = auto_increment(args.step, args.all)
### Step #5 - Compile all steps into one function: Skip-gram Sampling table
if args.step == 5: 
    print("\n### Step #5 - Compile all steps into one function: Skip-gram Sampling table")

    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(size=10)
    print(sampling_table)


logger.info('To be continued...')



### End of File
if args.plot:
    plt.show()
debug()


