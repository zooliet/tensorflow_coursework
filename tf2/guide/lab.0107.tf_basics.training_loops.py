#!/usr/bin/env python

import sys
sys.path.append('./')
sys.path.append('../../')

from lab_utils import (
    tf, os, np, plt, logger, ap, BooleanAction,
    debug, toc, auto_increment
)

# ap.add_argument('--epochs', type=int, default=10, help='number of epochs: 10*')
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
import timeit
import datetime


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - Solving machine learning problems
if args.step == 1:
    print("\n### Step #1 - Solving machine learning problems")

    __doc__='''
    - Obtain training data.
    - Define the model.
    - Define a loss function.
    - Run through the training data, calculating loss from the ideal value
    - Calculate gradients for that loss and use an optimizer to adjust the
      variables to fit the data.
    - Evaluate your results.
    '''
    print(__doc__)
    

args.step = auto_increment(args.step, args.all)
### Step #2 - Data
if args.step >= 2:
    print("\n### Step #2 - Data")

    # The actual line
    TRUE_W = 3.0
    TRUE_B = 2.0

    NUM_EXAMPLES = 1000

    # A vector of random x values
    x = tf.random.normal(shape=[NUM_EXAMPLES])

    # Generate some noise
    noise = tf.random.normal(shape=[NUM_EXAMPLES])

    # Calculate y
    y = x * TRUE_W + TRUE_B + noise

    if args.step == 2 and args.plot:
        plt.figure()
        plt.scatter(x, y, c="b")
        plt.show(block=False)

    __doc__='''
    Tensors are usually gathered together in batches, or groups of inputs and
    outputs stacked together. Batching can confer some training benefits and
    works well with accelerators and vectorized computation.  Given how small
    this dataset is, you can treat the entire dataset as a single batch.
    '''
    if args.step == 2: print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #3 - Define the model
if args.step >= 3:
    print("\n### Step #3 - Define the model")

    class MyModel(tf.Module):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Initialize the weights to `5.0` and the bias to `0.0`
            # In practice, these should be randomly initialized
            self.w = tf.Variable(5.0)
            self.b = tf.Variable(0.0)

        def __call__(self, x):
            return self.w * x + self.b

    model = MyModel()

    if args.step == 3:
        # List the variables tf.modules's built-in variable aggregation.
        logger.info("Variables:")
        print(*model.variables, sep='\n')
        print('')

    # Verify the model works
    assert model(3.0).numpy() == 15.0


args.step = auto_increment(args.step, args.all)
### Step #4 - Define the model: Define a loss function
if args.step >= 4:
    print("\n### Step #4 - Define the model: Define a loss function")

    __doc__='''
    A loss function measures how well the output of a model for a given input
    matches the target output. The goal is to minimize this difference during
    training. Define the standard L2 loss, also known as the "mean squared"
    error
    '''
    if args.step == 4: print(__doc__)

    # This computes a single loss value for an entire batch
    def loss(target_y, predicted_y):
        return tf.reduce_mean(tf.square(target_y - predicted_y))

    if args.step == 4:
        logger.info("Current loss: %1.3f\n" % loss(y, model(x)).numpy())

        # Before training the model, you can visualize the loss value by plotting 
        # the model's predictions in red and the training data in blue
        if args.plot:
            plt.figure()
            plt.scatter(x, y, c="b")
            plt.scatter(x, model(x), c="r")
            plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #5 - Define the model: Define a training loop
if args.step >= 5:
    print("\n### Step #5 - Define the model: Define a training loop")

    __doc__='''
    The training loop consists of repeatedly doing three tasks in order:
    - Sending a batch of inputs through the model to generate outputs
    - Calculating the loss by comparing the outputs to the output (or label)
    - Using gradient tape to find the gradients
    - Optimizing the variables with those gradients
    '''
    if args.step == 5: print(__doc__)

    # Given a callable model, inputs, outputs, and a learning rate...
    def train(model, x, y, learning_rate):
        with tf.GradientTape() as t:
            # Trainable variables are automatically tracked by GradientTape
            current_loss = loss(y, model(x))

        # Use GradientTape to calculate the gradients with respect to W and b
        dw, db = t.gradient(current_loss, [model.w, model.b])

        # Subtract the gradient scaled by the learning rate
        model.w.assign_sub(learning_rate * dw)
        model.b.assign_sub(learning_rate * db)

    # Collect the history of W-values and b-values to plot later
    Ws, bs = [], []
    epochs = range(10)

    # Define a training loop
    def training_loop(model, x, y):
        for epoch in epochs:
            # Update the model with the single giant batch
            train(model, x, y, learning_rate=0.1)

            # Track this before I update
            Ws.append(model.w.numpy())
            bs.append(model.b.numpy())
            current_loss = loss(y, model(x))

            print("Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f" % 
                    (epoch, Ws[-1], bs[-1], current_loss))
        print('')

    if args.step == 5:
        # For a look at training, you can send the same batch of x and y 
        # through the training loop, and see how W and b evolve.
        model = MyModel()

        print("Starting: W=%1.2f b=%1.2f, loss=%2.5f" % 
                (model.w, model.b, loss(y, model(x))))

        # Do the training
        training_loop(model, x, y)
        logger.info("Current loss: %1.5f\n" % loss(model(x), y).numpy())

        if args.plot:
            plt.figure()
            plt.plot(epochs, Ws, "r", epochs, bs, "b")
            plt.plot([TRUE_W] * len(epochs), "r--", [TRUE_B] * len(epochs), "b--")
            plt.legend(["W", "b", "True W", "True b"])
            plt.show(block=False)

            # Visualize how the trained model performs
            plt.figure()
            plt.scatter(x, y, c="b")
            plt.scatter(x, model(x), c="r")
            plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #6 - The same solution, but with Keras
if args.step == 6:
    print("\n### Step #6 - The same solution, but with Keras")

    class MyModelKeras(tf.keras.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Initialize the weights to `5.0` and the bias to `0.0`
            # In practice, these should be randomly initialized
            self.w = tf.Variable(5.0)
            self.b = tf.Variable(0.0)

        def call(self, x):
            return self.w * x + self.b

    keras_model = MyModelKeras()

    logger.info('Reuse the training loop with a Keras model:')
    training_loop(keras_model, x, y)

    # You can also save a checkpoint using Keras's built-in support
    keras_model.save_weights("tmp/tf2_g0107/my_checkpoint")
        
    keras_model = MyModelKeras()

    # compile sets the training parameters
    keras_model.compile(
        # By default, fit() uses tf.function().  You can
        # turn that off for debugging, but it is on now.
        run_eagerly=False,
        # Using a built-in optimizer, configuring as an object
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
        # Keras comes with built-in MSE error
        # However, you could use the loss function
        # defined above
        loss=tf.keras.losses.mean_squared_error,
    )

    # Keras fit expects batched data or a complete dataset as a NumPy array. 
    # NumPy arrays are chopped into batches and default to a batch size of 32.
    # In this case, to match the behavior of the hand-written loop, you should pass x in 
    # as a single batch of size 1000.
    logger.info('Use the built-in fit() of a Keras model:')
    keras_model.fit(x, y, epochs=10, batch_size=1000, verbose=2)
    print('')


### End of File
if args.plot:
    plt.show()
debug()
