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
### Step #1 - Overview: What are graphs?
if args.step == 1:
    print("\n### Step #1 - Overview: What are graphs?")

    __doc__='''
    In the previous three guides, you ran TensorFlow eagerly. This means
    TensorFlow operations are executed by Python, operation by operation, and
    returning results back to Python.

    While eager execution has several unique advantages, graph execution
    enables portability outside Python and tends to offer better performance.
    Graph execution means that tensor computations are executed as a TensorFlow
    graph, sometimes referred to as a tf.Graph or simply a "graph."

    Graphs are data structures that contain a set of tf.Operation objects,
    which represent units of computation; and tf.Tensor objects, which
    represent the units of data that flow between operations. They are defined
    in a tf.Graph context. Since these graphs are data structures, they can be
    saved, run, and restored all without the original Python code.
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #2 - The benefits of graphs
if args.step == 2:
    print("\n### Step #2 - The benefits of graphs")

    __doc__='''
    With a graph, you have a great deal of flexibility. You can use your
    TensorFlow graph in environments that don't have a Python interpreter, like
    mobile applications, embedded devices, and backend servers. TensorFlow uses
    graphs as the format for saved models when it exports them from Python.

    In short, graphs are extremely useful and let your TensorFlow run fast, run
    in parallel, and run efficiently on multiple devices.

    However, you still want to define your machine learning models (or other
    computations) in Python for convenience, and then automatically construct
    graphs when you need them.
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #3 - Taking advantage of graphs
if args.step == 3:
    print("\n### Step #3 - Taking advantage of graphs")

    __doc__='''
    You create and run a graph in TensorFlow by using tf.function, either as a
    direct call or as a decorator. tf.function takes a regular function as
    input and returns a Function. A Function is a Python callable that builds
    TensorFlow graphs from the Python function. You use a Function in the same
    way as its Python equivalent.
    '''
    print(__doc__)
    
    # Define a Python function.
    def a_regular_function(x, y, b):
      x = tf.matmul(x, y)
      x = x + b
      return x

    # `a_function_that_uses_a_graph` is a TensorFlow `Function`.
    a_function_that_uses_a_graph = tf.function(a_regular_function)

    # Make some tensors.
    x1 = tf.constant([[1.0, 2.0]])
    y1 = tf.constant([[2.0], [3.0]])
    b1 = tf.constant(4.0)

    orig_value = a_regular_function(x1, y1, b1).numpy()
    # Call a `Function` like a Python function.
    tf_function_value = a_function_that_uses_a_graph(x1, y1, b1).numpy()
    assert(orig_value == tf_function_value)

    __doc__='''
    On the outside, a Function looks like a regular function you write using
    TensorFlow operations. Underneath, however, it is very different. A
    Function encapsulates several tf.Graphs behind one API. That is how
    Function is able to give you the benefits of graph execution, like speed
    and deployability.
    '''
    print(__doc__)

    def inner_function(x, y, b):
        x = tf.matmul(x, y)
        x = x + b
        return x

    # Use the decorator to make `outer_function` a `Function`.
    @tf.function
    def outer_function(x):
        y = tf.constant([[2.0], [3.0]])
        b = tf.constant(4.0)

        return inner_function(x, y, b)

    # Note that the callable will create a graph that
    # includes `inner_function` as well as `outer_function`.
    logger.info('using @tf.function decorator:')
    result = outer_function(tf.constant([[1.0, 2.0]])).numpy()
    print(result)


args.step = auto_increment(args.step, args.all)
### Step #4 - Taking advantage of graphs: Converting Python functions to graphs
if args.step == 4:
    print("\n### Step #4 - Taking advantage of graphs: Converting Python functions to graphs")

    def simple_relu(x):
        if tf.greater(x, 0):
            return x
        else:
            return 0

    logger.info('tf_simple_relu is a TensorFlow Function that wraps simple_relu:')
    tf_simple_relu = tf.function(simple_relu)

    print("First branch, with graph:", tf_simple_relu(tf.constant(1)).numpy())
    print("Second branch, with graph:", tf_simple_relu(tf.constant(-1)).numpy())
    print()

    logger.info('This is the graph-generating output of AutoGraph:')
    print(tf.autograph.to_code(simple_relu))

    logger.info('This is the graph itself:')
    print(tf_simple_relu.get_concrete_function(tf.constant(1)).graph.as_graph_def())


args.step = auto_increment(args.step, args.all)
### Step #5 - Taking advantage of graphs: Polymorphism: one Function, many graphs
if args.step == 5:
    print("\n### Step #5 - Taking advantage of graphs: Polymorphism: one Function, many graphs")

    __doc__='''
    Each time you invoke a Function with new dtypes and shapes in its
    arguments, Function creates a new tf.Graph for the new arguments. The
    dtypes and shapes of a tf.Graph's inputs are known as an input signature or
    just a signature.

    The Function stores the tf.Graph corresponding to that signature in a
    ConcreteFunction.  A ConcreteFunction is a wrapper around a tf.Graph
    '''
    print(__doc__)

    @tf.function
    def my_relu(x):
        return tf.maximum(0., x)

    logger.info('my_relu() creates new graphs as it observes more signatures:')
    print(my_relu(tf.constant(5.5)))
    print(my_relu([1, -1]))
    print(my_relu(tf.constant([3., -3.])), '\n')

    logger.info('If the Function has already been called with that signature, Function does not create a new tf.Graph:')
    # These two calls do *not* create new graphs.
    print(my_relu(tf.constant(-2.5))) # Signature matches `tf.constant(5.5)`.
    print(my_relu(tf.constant([-1., 1.])), '\n') # Signature matches `tf.constant([3., -3.])`.

    logger.info('There are three ConcreteFunctions (one for each graph) in my_relu:')
    # The `ConcreteFunction` also knows the return type and shape!
    print(my_relu.pretty_printed_concrete_signatures())


args.step = auto_increment(args.step, args.all)
### Step #6 - Using tf.function: Graph execution vs. eager execution
if args.step == 6:
    print("\n### Step #6 - Graph execution vs. eager execution")

    @tf.function
    def get_MSE(y_true, y_pred):
        print("Calculating MSE!")
        sq_diff = tf.pow(y_true - y_pred, 2)
        return tf.reduce_mean(sq_diff)

    y_true = tf.random.uniform([5], maxval=10, dtype=tf.int32)
    y_pred = tf.random.uniform([5], maxval=10, dtype=tf.int32)

    logger.info('By default, Function executes its code as a graph:')
    get_MSE(y_true, y_pred)
    get_MSE(y_true, y_pred)
    get_MSE(y_true, y_pred)

    __doc__='''
    get_MSE only printed once even though it was called three times.

    To explain, the print statement is executed when Function runs the original
    code in order to create the graph in a process known as "tracing". Tracing
    captures the TensorFlow operations into a graph, and print is not captured
    in the graph.  That graph is then executed for all three calls without ever
    running the Python code again.

    If you would like to print values in both eager and graph execution, use
    tf.print instead.
    '''
    print(__doc__)

    logger.info('As a sanity check, let\'s turn off graph execution to compare:')
    # Now, globally set everything to run eagerly to force eager execution.
    tf.config.run_functions_eagerly(True)

    # # Observe what is printed below.
    get_MSE(y_true, y_pred)
    get_MSE(y_true, y_pred)
    get_MSE(y_true, y_pred)

    # Don't forget to set it back when you are done
    tf.config.run_functions_eagerly(False)


args.step = auto_increment(args.step, args.all)
### Step #7 - Using tf.function: tf.function best practices
if args.step == 7:
    print("\n### Step #7 - Using tf.function: tf.function best practices")

    __doc__='''
    - Toggle between eager and graph execution early and often with
      tf.config.run_functions_eagerly to pinpoint if/ when the two modes
      diverge.
    - Create tf.Variables outside the Python function and modify them on the
      inside.  The same goes for objects that use tf.Variable, like
      keras.layers, keras.Models and tf.optimizers.
    - Avoid writing functions that depend on outer Python variables, excluding
      tf.Variables and Keras objects.
    - Prefer to write functions which take tensors and other TensorFlow types
      as input.  You can pass in other object types but be careful!
    - Include as much computation as possible under a tf.function to maximize
      the performance gain. For example, decorate a whole training step or the
      entire training loop
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #8 - Seeing the speed-up
if args.step == 8:
    print("\n### Step #8 - Seeing the speed-up")
    
    __doc__='''
    tf.function usually improves the performance of your code, but the amount
    of speed-up depends on the kind of computation you run. Small computations
    can be dominated by the overhead of calling a graph.
    '''
    print(__doc__)

    logger.info('measure the difference in performance:')
    x = tf.random.uniform(shape=[10, 10], minval=-1, maxval=2, dtype=tf.dtypes.int32)

    def power(x, y):
        result = tf.eye(10, dtype=tf.dtypes.int32)
        for _ in range(y):
            result = tf.matmul(x, result)
        return result

    print("Eager execution:", timeit.timeit(lambda: power(x, 100), number=1000))
    
    power_as_graph = tf.function(power)
    print("Graph execution:", timeit.timeit(lambda: power_as_graph(x, 100), number=1000))


args.step = auto_increment(args.step, args.all)
### Step #9 - Seeing the speed-up: Performance and trade-offs
if args.step == 9:
    print("\n### Step #9 - Seeing the speed-up: Performance and trade-offs")

    __doc__='''
    Graphs can speed up your code, but the process of creating them has some
    overhead.  For some functions, the creation of the graph takes more time
    than the execution of the graph. This investment is usually quickly paid
    back with the performance boost of subsequent executions, but it's
    important to be aware that the first few steps of any large model training
    can be slower due to tracing.

    No matter how large your model, you want to avoid tracing frequently. The
    tf.function guide discusses how to set input specifications and use tensor
    arguments to avoid retracing. If you find you are getting unusually poor
    performance, it's a good idea to check if you are retracing accidentally.
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #10 - When is a Function tracing
if args.step == 10:
    print("\n### Step #10 - When is a Function tracing")

    __doc__ = '''
    To figure out when your Function is tracing, add a print statement to its
    code. As a rule of thumb, Function will execute the print statement every
    time it traces.   
    '''
    print(__doc__)

    @tf.function
    def a_function_with_python_side_effect(x):
        print("Tracing!") # An eager-only side effect.
        return x * x + tf.constant(2)

    logger.info('This is traced the first time:')
    print(a_function_with_python_side_effect(tf.constant(2)), '\n')

    logger.info('The second time through, you won\'t see the side effect:')
    print(a_function_with_python_side_effect(tf.constant(3)), '\n')

    str='''This retraces each time the Python argument changes,
    \ras a Python argument could be an epoch count or other hyperparameter:'''
    logger.info(str)
    print(a_function_with_python_side_effect(2))
    print(a_function_with_python_side_effect(3))

    __doc__='''
    New Python arguments always trigger the creation of a new graph, hence the
    extra tracing.
    '''
    print(__doc__)


### End of File
print('')
if args.plot:
    plt.show()
debug()
