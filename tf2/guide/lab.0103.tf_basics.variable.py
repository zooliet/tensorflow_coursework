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


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - Create a variable
if args.step == 1:
    print("\n### Step #1 - Create a variable")

    logger.info('A variable looks and acts like a tensor:')
    my_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    my_variable = tf.Variable(my_tensor)

    # Variables can be all kinds of types, just like tensors
    bool_variable = tf.Variable([False, False, False, True])
    complex_variable = tf.Variable([5 + 4j, 6 + 1j])

    print("Shape: ", my_variable.shape)
    print("DType: ", my_variable.dtype)
    print("As NumPy:\n", my_variable.numpy(), '\n')  

    logger.info('Most tensor ops work on variables as expected:')
    print("A variable:\n", my_variable)
    print("Viewed as a tensor:\n", tf.convert_to_tensor(my_variable))
    print("Index of highest value:\n", tf.argmax(my_variable), '\n')

    logger.info('This creates a new tensor; it does not reshape the variable:')
    print("Copying and reshaping:\n", tf.reshape(my_variable, [1,4]), '\n')

    a = tf.Variable([2.0, 3.0])
    # This will keep the same dtype, float32
    a.assign([1, 2])
    # Not allowed as it resizes the variable:
    try:
        a.assign([1.0, 2.0, 3.0])
    except Exception as e:
        print(f"{type(e).__name__}: {e}\n")

    logger.info('Create b based on the value of a: a and b are different:')
    a = tf.Variable([2.0, 3.0])
    b = tf.Variable(a)
    a.assign([5, 6])

    # a and b are different
    print(a.numpy())
    print(b.numpy())
    print('')

    logger.info('There are other versions of assign:')
    print(a.assign_add([2,3]).numpy())  # [7. 9.]
    print(a.assign_sub([7,9]).numpy())  # [0. 0.]
    print('')


args.step = auto_increment(args.step, args.all)
### Step #2 - Lifecycles, naming, and watching
if args.step == 2:
    print("\n### Step #2 - Lifecycles, naming, and watching")

    str = '''
    In Python-based TensorFlow, tf.Variable instance have the same lifecycle as other Python 
    objects. When there are no references to a variable it is automatically deallocated.
    '''
    print(str)

    logger.info('Variables can be named which can help you track and debug them:')
    # Create a and b; they will have the same name but will be backed by different tensors.
    my_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    a = tf.Variable(my_tensor, name="Mark")
    # A new variable with the same name, but different value
    # Note that the scalar add is broadcast
    b = tf.Variable(my_tensor + 1, name="Mark")

    # These are elementwise-unequal, despite having the same name
    print(a == b)

    str='''
    Variable names are preserved when saving and loading models. By default, variables in 
    models will acquire unique variable names automatically, so you don't need to assign 
    them yourself unless you want to.
    Although variables are important for differentiation, some variables will not need to 
    be differentiated. You can turn off gradients for a variable by setting trainable to 
    false at creation. An example of a variable that would not need gradients is a training 
    step counter.
    '''
    print(str)
    step_counter = tf.Variable(1, trainable=False)


args.step = auto_increment(args.step, args.all)
### Step #3 - Indexing: Placing variables and tensors
if args.step == 3:
    print("\n### Step #3 - Placing variables and tensors")

    # Uncomment to see where your variables get placed (see below)
    # tf.debugging.set_log_device_placement(True)

    with tf.device('CPU:0'):
        # Create some tensors 
        a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print(c, '\n')

    with tf.device('CPU:0'):
        a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.Variable([[1.0, 2.0, 3.0]])

    with tf.device('GPU:0'):
        # Element-wise multiply
        k = a * b
    print(k, '\n')


### End of File
if args.plot:
    plt.show()
debug()
