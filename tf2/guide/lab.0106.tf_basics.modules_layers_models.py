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
### Step #1 - Defining models and layers in TensorFlow
if args.step == 1:
    print("\n### Step #1 - Defining models and layers in TensorFlow")

    __doc__='''
    A model is, abstractly:
    - A function that computes something on tensors (a forward pass)
    - Some variables that can be updated in response to training

    Most models are made of layers. Layers are functions with a known
    mathematical structure that can be reused and have trainable variables. In
    TensorFlow, most high-level implementations of layers and models, such as
    Keras or Sonnet, are built on the same foundational class: tf.Module.
    '''
    print(__doc__)

    class SimpleModule(tf.Module):
        def __init__(self, name=None):
            super().__init__(name=name)
            self.a_variable = tf.Variable(5.0, name="train_me")
            self.non_trainable_variable = tf.Variable(5.0, trainable=False, name="do_not_train_me")
        def __call__(self, x):
            return self.a_variable * x + self.non_trainable_variable

    simple_module = SimpleModule(name="simple")
    logger.info('a very simple tf.Module on a scaler tensor:')
    print(simple_module(tf.constant(5.0)), '\n')

    # All trainable variables
    simple_module.trainable_variables
    print()
    print("trainable variables:")
    print(*simple_module.trainable_variables, sep='\n')
    print()
    # Every variable
    print("all variables:")
    print(*simple_module.variables, sep='\n')
    print()

    # example of a two-layer linear layer model made out of modules
    class Dense(tf.Module):
        def __init__(self, in_features, out_features, name=None):
            super().__init__(name=name)
            self.w = tf.Variable(tf.random.normal([in_features, out_features]), name='w')
            self.b = tf.Variable(tf.zeros([out_features]), name='b')
        def __call__(self, x):
            y = tf.matmul(x, self.w) + self.b
            return tf.nn.relu(y)    

    class SequentialModule(tf.Module):
        def __init__(self, name=None):
            super().__init__(name=name)
            self.dense_1 = Dense(in_features=3, out_features=3)
            self.dense_2 = Dense(in_features=3, out_features=2)

        def __call__(self, x):
            x = self.dense_1(x)
            return self.dense_2(x)

    # You have made a model!
    my_model = SequentialModule(name="the_model")

    # Call it, with random results
    logger.info('Two-layer linear model results:')
    print(my_model(tf.constant([[2.0, 2.0, 2.0]])))

    __doc__='''
    tf.Module instances will automatically collect, recursively, any
    tf.Variable or tf.Module instances assigned to it. This allows you to
    manage collections of tf.Modules with a single model instance, and save and
    load whole models.
    '''
    print(__doc__)

    logger.info('Submodules:') 
    print(*my_model.submodules, sep='\n')
    print()

    logger.info('Varaiables:')
    print(*my_model.variables, sep='\n\n')


args.step = auto_increment(args.step, args.all)
### Step #2 - Defining models and layers in TensorFlow: Waiting to create variables
if args.step in [2, 3]:
    print("\n### Step #2 - Defining models and layers in TensorFlow: Waiting to create variables")

    __doc__='''
    You may have noticed here that you have to define both input and output
    sizes to the layer. This is so the w variable has a known shape and can be
    allocated.

    By deferring variable creation to the first time the module is called with
    a specific input shape, you do not need specify the input size up front.

    This flexibility is why TensorFlow layers often only need to specify the
    shape of their outputs, such as in tf.keras.layers.Dense, rather than both
    the input and output size.
    '''
    if args.step == 2: print(__doc__)

    class FlexibleDenseModule(tf.Module):
        # Note: No need for `in_features`
        def __init__(self, out_features, name=None):
            super().__init__(name=name)
            self.is_built = False
            self.out_features = out_features

        def __call__(self, x):
            # Create variables on first call.
            if not self.is_built:
                self.w = tf.Variable(tf.random.normal([x.shape[-1], self.out_features]), name='w')
                self.b = tf.Variable(tf.zeros([self.out_features]), name='b')
                self.is_built = True

            y = tf.matmul(x, self.w) + self.b
            return tf.nn.relu(y)
        
    # Used in a module
    class MySequentialModule(tf.Module):
        def __init__(self, name=None):
            super().__init__(name=name)
            self.dense_1 = FlexibleDenseModule(out_features=3)
            self.dense_2 = FlexibleDenseModule(out_features=2)

        def __call__(self, x):
            x = self.dense_1(x)
            return self.dense_2(x)

    my_model = MySequentialModule(name="the_model")
    result = my_model(tf.constant([[2.0, 2.0, 2.0]]))

    if args.step == 2:
        logger.info(f'Model results: {result}')


args.step = auto_increment(args.step, args.all)
### Step #3 - Saving weights
if args.step == 3:
    print("\n### Step #3 - Saving weights")

    __doc__='''
    You can save a tf.Module as both a checkpoint and a SavedModel.
    Checkpoints are just the weights (that is, the values of the set of
    variables inside the module and its submodules):
    '''
    print(__doc__)
    
    chkp_path = "tmp/tf2_g0106/my_checkpoint"
    checkpoint = tf.train.Checkpoint(model=my_model)
    checkpoint.write(chkp_path)

    logger.info("tf.io.gfile.glob(chkp_path+'*')")
    print(*tf.io.gfile.glob(chkp_path+'*'), sep='\n')
    print()

    logger.info('the whole collection of variables is saved inside the checkpoint:')
    print(*tf.train.list_variables(chkp_path), sep='\n')
    print()

    logger.info('When you load models back in:')
    new_model = MySequentialModule()
    new_checkpoint = tf.train.Checkpoint(model=new_model)
    new_checkpoint.restore("tmp/tf2_g0106/my_checkpoint")

    # Should be the same result as above
    result = new_model(tf.constant([[2.0, 2.0, 2.0]]))
    print(result)


args.step = auto_increment(args.step, args.all)
### Step #4 - Saving functions
if args.step in [4, 5]:
    print("\n### Step #4 - Saving functions")

    __doc__='''
    TensorFlow can run models without the original Python objects, as
    demonstrated by TensorFlow Serving and TensorFlow Lite, even when you
    download a trained model from TensorFlow Hub.

    TensorFlow needs to know how to do the computations described in Python,
    but without the original code. To do this, you can make a graph, which is
    described in the Introduction to graphs and functions guide.

    This graph contains operations, or ops, that implement the function.

    You can define a graph in the model above by adding the @tf.function
    decorator to indicate that this code should run as a graph.
    '''
    if args.step == 4: print(__doc__)

    class Dense(tf.Module):
        def __init__(self, in_features, out_features, name=None):
            super().__init__(name=name)
            self.w = tf.Variable(tf.random.normal([in_features, out_features]), name='w')
            self.b = tf.Variable(tf.zeros([out_features]), name='b')
        def __call__(self, x):
            y = tf.matmul(x, self.w) + self.b
            return tf.nn.relu(y)    

    class MySequentialModule(tf.Module):
        def __init__(self, name=None):
            super().__init__(name=name)
            self.dense_1 = Dense(in_features=3, out_features=3)
            self.dense_2 = Dense(in_features=3, out_features=2)

        @tf.function
        def __call__(self, x):
            x = self.dense_1(x)
            return self.dense_2(x)

    # You have made a model with a graph!
    my_model = MySequentialModule(name="the_model")

    if args.step == 4:
        logger.info('Test a model built using tf.function:')
        print(my_model([[2.0, 2.0, 2.0]]), '\n')
        # print(my_model([[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]), '\n')

    # You can visualize the graph by tracing it within a TensorBoard summary
    # Set up logging.
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = "tmp/tf2_g0106/logs/func/%s" % stamp
    writer = tf.summary.create_file_writer(logdir)

    # Create a new model to get a fresh trace
    # Otherwise the summary will not see the graph.
    new_model = MySequentialModule()

    # Bracket the function call with
    # tf.summary.trace_on() and tf.summary.trace_export().
    tf.summary.trace_on(graph=True)
    tf.profiler.experimental.start(logdir)
    # Call only one tf.function when tracing.
    z = print(new_model(tf.constant([[2.0, 2.0, 2.0]])), '- for tracing')
    with writer.as_default():
        tf.summary.trace_export(
            name="my_func_trace",
            step=0,
            profiler_outdir=logdir
        )

    if args.step == 4:
        logger.info('Launch TensorBoard to view the resulting trace:')
        print('tensorboard --logdir tmp/tf2_g0106/logs/func --bind_all') 


args.step = auto_increment(args.step, args.all)
### Step #5 - Saving functions: Creating a SavedModel
if args.step == 5:
    print("\n### Step #5 - Saving functions: Creating a SavedModel")
    
    __doc__='''
    The recommended way of sharing completely trained models is to use
    SavedModel.  SavedModel contains both a collection of functions and a
    collection of weights.
    '''
    print(__doc__)

    logger.info('You can save the model you have just trained as follows:')
    tf.saved_model.save(my_model, "tmp/tf2_g0106/the_saved_model")

    logger.info('ls -l tmp/tf2_g0106/the_saved_model:')
    print(*os.listdir('tmp/tf2_g0106/the_saved_model'), sep='\n')
    print()

    # The variables/ directory contains a checkpoint of the variables
    logger.info('ls -l the_saved_model/tf2_g0106/variables:')
    print(*os.listdir('tmp/tf2_g0106/the_saved_model/variables'), sep='\n')

    __doc__='''
    The saved_model.pb file is a protocol buffer describing the functional
    tf.Graph

    Models and layers can be loaded from this representation without actually
    making an instance of the class that created it. This is desired in
    situations where you do not have (or want) a Python interpreter, such as
    serving at scale or on an edge device, or in situations where the original
    Python code is not available or practical to use.

    You can load the model as new object: tf.saved_model.load()

    new_model, created from loading a saved model, is an internal TensorFlow
    user object without any of the class knowledge.  
    '''
    print(__doc__)

    new_model = tf.saved_model.load("tmp/tf2_g0106/the_saved_model")
    logger.info(f'It is not of type MySequentialModule: {isinstance(new_model, MySequentialModule)}')


args.step = auto_increment(args.step, args.all)
### Step #6 - Keras models and layers: Keras layers
if args.step == 6:
    print("\n### Step #6 - Keras models and layers: Keras layers")
    
    __doc__='''
    tf.keras.layers.Layer is the base class of all Keras layers, and it
    inherits from tf.Module. You can convert a module into a Keras layer just
    by swapping out the parent and then changing __call__ to call
    '''
    print(__doc__)

    class MyDense(tf.keras.layers.Layer):
        # Adding **kwargs to support base Keras layer arguments
        def __init__(self, in_features, out_features, **kwargs):
            super().__init__(**kwargs)

            # This will soon move to the build step; see below
            self.w = tf.Variable(tf.random.normal([in_features, out_features]), name='w')
            self.b = tf.Variable(tf.zeros([out_features]), name='b')

        def call(self, x):
            y = tf.matmul(x, self.w) + self.b
            return tf.nn.relu(y)

    simple_layer = MyDense(name="simple", in_features=3, out_features=3)
    logger.info('simple_layer([[2.0, 2.0, 2.0]]:')
    print(simple_layer([[2.0, 2.0, 2.0]]))


args.step = auto_increment(args.step, args.all)
### Step #7 - Keras models and layers: The build step
if args.step in [7, 8, 9]:
    print("\n### Step #7 - Keras models and layers: The build step")

    __doc__='''
    As noted, it's convenient in many cases to wait to create variables until
    you are sure of the input shape. Keras layers come with an extra lifecycle
    step that allows you more flexibility in how you define your layers. This
    is defined in the build function.
    
    build is called exactly once, and it is called with the shape of the input.
    It's usually used to create variables (weights).
    '''
    if args.step == 7: print(__doc__)

    class FlexibleDense(tf.keras.layers.Layer):
        # Note the added `**kwargs`, as Keras supports many arguments
        def __init__(self, out_features, **kwargs):
            super().__init__(**kwargs)
            self.out_features = out_features

        def build(self, input_shape):  # Create the state of the layer (weights)
            self.w = tf.Variable(tf.random.normal([input_shape[-1], self.out_features]), name='w')
            self.b = tf.Variable(tf.zeros([self.out_features]), name='b')

        def call(self, inputs):  # Defines the computation from inputs to outputs
            return tf.matmul(inputs, self.w) + self.b

    # Create the instance of the layer
    flexible_dense = FlexibleDense(out_features=3)

    if args.step == 7:
        logger.info('At this point, the model has not been built, so there are no variables:')
        print(flexible_dense.variables, '\n')

        logger.info('Calling the function allocates appropriately-sized variables:')
        # Call it, with predictably random results
        print(flexible_dense(tf.constant([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])), '\n')

        logger.info('flexible_dense.variables:')
        print(*flexible_dense.variables, sep='\n\n')

    __doc__='''
    Since build is only called once, inputs will be rejected if the input shape
    is not compatible with the layer's variables:
    '''
    if args.step == 7: 
        print(__doc__)

        try:
            print("Model results:", flexible_dense(tf.constant([[2.0, 2.0, 2.0, 2.0]])))
        except tf.errors.InvalidArgumentError as e:
            print("Failed:", e)


args.step = auto_increment(args.step, args.all)
### Step #8 - Keras models and layers: Keras models
if args.step in [8, 9]:
    print("\n### Step #8 - Keras models and layers: Keras models")

    __doc__='''
    Keras also provides a full-featured model class called tf.keras.Model. It
    inherits from tf.keras.layers.Layer, so a Keras model can be used, nested,
    and saved in the same way as Keras layers. Keras models come with extra
    functionality that makes them easy to train, evaluate, load, save, and even
    train on multiple machines.
    '''
    if args.step == 8: print(__doc__)

    class MySequentialModel(tf.keras.Model):
        def __init__(self, name=None, **kwargs):
            super().__init__(**kwargs)
            self.dense_1 = FlexibleDense(out_features=3)
            self.dense_2 = FlexibleDense(out_features=2)
        def call(self, x):
            x = self.dense_1(x)
            return self.dense_2(x)

    # You have made a Keras model!
    my_sequential_model = MySequentialModel(name="the_model")
    result = my_sequential_model(tf.constant([[2.0, 2.0, 2.0]])),

    if args.step == 8:
        # Call it on a tensor, with random results
        logger.info(f"Model results:\n{result}\n")

        logger.info('my_sequential_model\'s variables and submodules:')
        print(*my_sequential_model.variables, sep='\n\n')
        print()
        print(*my_sequential_model.submodules, sep='\n')

    __doc__='''
    Overriding tf.keras.Model is a very Pythonic approach to building
    TensorFlow models.  If you are migrating models from other frameworks, this
    can be very straightforward.

    If you are constructing models that are simple assemblages of existing
    layers and inputs, you can save time and space by using the functional API,
    which comes with additional features around model reconstruction and
    architecture.
    '''
    if args.step == 8: print(__doc__)

    inputs = tf.keras.Input(shape=[3,])

    x = FlexibleDense(3)(inputs)
    x = FlexibleDense(2)(x)

    if args.step == 8:
        my_functional_model = tf.keras.Model(inputs=inputs, outputs=x)
        logger.info('functional api model summary:')
        my_functional_model.summary()
        print()

        logger.info('functional api model result:')
        print(my_functional_model(tf.constant([[2.0, 2.0, 2.0]])))


args.step = auto_increment(args.step, args.all)
### Step #9 - Saving Keras models
if args.step == 9:
    print("\n### Step #9 - Saving Keras models")

    __doc__='''
    Keras models can be checkpointed, and that will look the same as tf.Module.
    Keras models can also be saved with tf.saved_model.save(), as they are
    modules.  However, Keras models have convenience methods and other
    functionality.
    '''
    print(__doc__)

    my_sequential_model.save("tmp/tf2_g0106/exname_of_file")
    reconstructed_model = tf.keras.models.load_model("tmp/tf2_g0106/exname_of_file")
    logger.info('reconstructed_model result:')
    print(reconstructed_model(tf.constant([[2.0, 2.0, 2.0]])))


### End of File
print()
if args.plot:
    plt.show()
debug()
