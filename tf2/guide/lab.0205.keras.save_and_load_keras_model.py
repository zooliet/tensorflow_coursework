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
from PIL import Image

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Layer, Dense, InputLayer, Dropout


### TOC
if args.step == 0:
    toc(__file__)


if args.step or args.all:
    def get_model():
        # Create a simple model.
        inputs = Input(shape=(32,))
        outputs = Dense(1)(inputs)
        model = Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model


args.step = auto_increment(args.step, args.all)
### Step #1 - Introduction
if args.step == 1:
    print("\n### Step #1 - Introduction")

    __doc__='''
    A Keras model consists of multiple components:
    - The architecture, or configuration, which specifies what layers the model
      contain, and how they're connected.
    - A set of weights values (the "state of the model").
    - An optimizer (defined by compiling the model).
    - A set of losses and metrics (defined by compiling the model or calling
      add_loss() or add_metric()).

    The Keras API makes it possible to save all of these pieces to disk at
    once, or to only selectively save some of them:
    - Saving everything into a single archive in the TensorFlow SavedModel
      format (or in the older Keras H5 format). This is the standard practice.
    - Saving the architecture / configuration only, typically as a JSON file.
    - Saving the weights values only. This is generally used when training the
      model.
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #2 - Whole-model saving & loading 
if args.step == 2:
    print("\n### Step #2 - Whole-model saving & loading")

    __doc__='''
    You can save an entire model to a single artifact. It will include:
    - The model's architecture/config
    - The model's weight values (which were learned during training)
    - The model's compilation information (if compile() was called)
    - The optimizer and its state, if any (this enables you to restart training
      where you left)

    There are two formats you can use to save an entire model to disk: the
    TensorFlow SavedModel format, and the older Keras H5 format. The
    recommended format is SavedModel.  It is the default when you use
    model.save().

    You can switch to the H5 format by:
    - Passing save_format='h5' to save().
    - Passing a filename that ends in .h5 or .keras to save().
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #3 - Whole-model saving & loading: SavedModel format
if args.step == 3:
    print("\n### Step #3 - Whole-model saving & loading: SavedModel format")
    model = get_model()
    
    # Train the model.
    test_input = np.random.random((128, 32))
    test_target = np.random.random((128, 1))
    model.fit(test_input, test_target, verbose=2)

    # Calling `save('my_model')` creates a SavedModel folder `my_model`.
    model.save("tmp/tf2_g0205/my_model")

    # It can be used to reconstruct the model identically.
    reconstructed_model = tf.keras.models.load_model("tmp/tf2_g0205/my_model")

    # Let's check:
    np.testing.assert_allclose(
        model.predict(test_input), reconstructed_model.predict(test_input)
    )

    # The reconstructed model is already compiled and has retained the optimizer
    # state, so training can resume:
    reconstructed_model.fit(test_input, test_target, verbose=2)

    __doc__='''
    The model architecture, and training configuration (including the
    optimizer, losses, and metrics) are stored in saved_model.pb. The weights
    are saved in the variables/ directory.
    '''
    print(__doc__)

    logger.info('ls -l .../my_model:')
    print(*os.listdir('tmp/tf2_g0205/my_model'), sep='\n')
    print()

    # How SavedModel handles custom objects
    class CustomModel(Model):
        def __init__(self, hidden_units):
            super(CustomModel, self).__init__()
            self.hidden_units = hidden_units
            self.dense_layers = [Dense(u) for u in hidden_units]

        def call(self, inputs):
            x = inputs
            for layer in self.dense_layers:
                x = layer(x)
            return x

        def get_config(self):
            return {"hidden_units": self.hidden_units}

        @classmethod
        def from_config(cls, config):
            return cls(**config)

    # Build the model by calling it
    model = CustomModel([16, 16, 10])

    input_arr = tf.random.uniform((1, 5))
    outputs = model(input_arr)
    model.save("tmp/tf2_g0205/my_model")

    # Option 1: Load with the custom_object argument.
    loaded_1 = tf.keras.models.load_model(
        "tmp/tf2_g0205/my_model", custom_objects={"CustomModel": CustomModel}
    )

    # Option 2: Load without the CustomModel class.

    # Delete the custom-defined model class to ensure that the loader does not have access to it.
    del CustomModel

    loaded_2 = tf.keras.models.load_model("tmp/tf2_g0205/my_model")
    np.testing.assert_allclose(loaded_1(input_arr), outputs)
    np.testing.assert_allclose(loaded_2(input_arr), outputs)

    logger.info("Original model:\n{}".format(model))
    logger.info("Model loaded with custom objects:\n{}".format(loaded_1))
    logger.info("Model loaded without the custom object class:\n{}".format(loaded_2))


args.step = auto_increment(args.step, args.all)
### Step #4 - Whole-model saving & loading: Keras H5 format
if args.step == 4:
    print("\n### Step #4 - Whole-model saving & loading: Keras H5 format")

    model = get_model()

    # Train the model.
    test_input = np.random.random((128, 32))
    test_target = np.random.random((128, 1))
    model.fit(test_input, test_target)

    # Calling `save('my_model.h5')` creates a h5 file `my_model.h5`.
    model.save("tmp/tf2_g0205/my_h5_model.h5")

    # It can be used to reconstruct the model identically.
    reconstructed_model = tf.keras.models.load_model("tmp/tf2_g0205/my_h5_model.h5")

    # Let's check:
    np.testing.assert_allclose(
        model.predict(test_input), reconstructed_model.predict(test_input)
    )

    # The reconstructed model is already compiled and has retained the optimizer
    # state, so training can resume:
    reconstructed_model.fit(test_input, test_target, verbose=2)


args.step = auto_increment(args.step, args.all)
### Step #5 - Saving the architecture
if args.step == 5:
    print("\n### Step #5 - Saving the architecture")

    __doc__='''
    The model's configuration (or architecture) specifies what layers the model
    contains, and how these layers are connected. If you have the configuration
    of a model, then the model can be created with a freshly initialized state
    for the weights and no compilation information.  Note: this only applies to
    models defined using the functional or Sequential apis not subclassed
    models.
    ''' 
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #6 - Saving the architecture: Configuration of a Sequential model or Functional API model
if args.step == 6:
    print("\n### Step #6 - Saving the architecture: Configuration of a Sequential model or Functional API model")

    __doc__='''
    Calling config = model.get_config() will return a Python dict containing
    the configuration of the model. The same model can then be reconstructed
    via Sequential.from_config(config) (for a Sequential model) or
    Model.from_config(config) (for a Functional API model).

    to_json() and tf.keras.models.model_from_json() is similar to get_config /
    from_config, except it turns the model into a JSON string, which can then
    be loaded without the original model class. It is also specific to models,
    it isn't meant for layers.
    '''
    print(__doc__)
    
    layer = Dense(3, activation="relu")
    layer_config = layer.get_config()
    new_layer = Dense.from_config(layer_config)

    # Sequential model example
    model = Sequential([InputLayer(input_shape=(32,)), Dense(1)])
    config = model.get_config()
    new_model = Sequential.from_config(config)

    # Functional model example
    inputs = Input((32,))
    outputs = Dense(1)(inputs)
    model = Model(inputs, outputs)
    config = model.get_config()
    new_model = Model.from_config(config)

    # json
    model = Sequential([InputLayer(input_shape=(32,)), Dense(1)])
    json_config = model.to_json()
    new_model = tf.keras.models.model_from_json(json_config)


args.step = auto_increment(args.step, args.all)
### Step #7 - Saving the architecture: Custom objects 
if args.step == 7:
    print("\n### Step #7 - Saving the architecture: Custom objects")
    pass


args.step = auto_increment(args.step, args.all)
### Step #8 - Saving the architecture: In-memory model cloning
if args.step == 8:
    print("\n### Step #8 - Saving the architecture: In-memory model cloning")
    pass


args.step = auto_increment(args.step, args.all)
### Step #9 - Saving & loading only the model's weights values 
if args.step == 9:
    print("\n### Step #9 - Saving & loading only the model's weights values")

    __doc__='''
    You can choose to only save & load a model's weights. 
    This can be useful if:
    - You only need the model for inference: in this case you won't need to
      restart training, so you don't need the compilation information or
      optimizer state.
    - You are doing transfer learning: in this case you will be training a new
      model reusing the state of a prior model, so you don't need the
      compilation information of the prior model.
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #10 - Saving & loading only the model's weights values: APIs for in-memory weight transfer
if args.step == 10:
    print("\n### Step #10 - Saving & loading only the model's weights values: APIs for in-memory weight transfer")

    # Transfering weights from one layer to another, in memory
    def create_layer():
        layer = Dense(64, activation="relu", name="dense_2")
        layer.build((None, 784))
        return layer

    layer_1 = create_layer()
    layer_2 = create_layer()

    # Copy weights from layer 1 to layer 2
    layer_2.set_weights(layer_1.get_weights()) 

    # Transfering weights from one model to another model with a compatible architecture, 
    # in memory

    # Create a simple functional model
    inputs = Input(shape=(784,), name="digits")
    x = Dense(64, activation="relu", name="dense_1")(inputs)
    x = Dense(64, activation="relu", name="dense_2")(x)
    outputs = Dense(10, name="predictions")(x)
    functional_model = Model(inputs=inputs, outputs=outputs, name="3_layer_mlp")

    # Define a subclassed model with the same architecture
    class SubclassedModel(Model):
        def __init__(self, output_dim, name=None):
            super(SubclassedModel, self).__init__(name=name)
            self.output_dim = output_dim
            self.dense_1 = Dense(64, activation="relu", name="dense_1")
            self.dense_2 = Dense(64, activation="relu", name="dense_2")
            self.dense_3 = Dense(output_dim, name="predictions")

        def call(self, inputs):
            x = self.dense_1(inputs)
            x = self.dense_2(x)
            x = self.dense_3(x)
            return x

        def get_config(self):
            return {"output_dim": self.output_dim, "name": self.name}

    subclassed_model = SubclassedModel(10)
    # Call the subclassed model once to create the weights.
    subclassed_model(tf.ones((1, 784)))

    # Copy weights from functional_model to subclassed_model.
    subclassed_model.set_weights(functional_model.get_weights())

    assert len(functional_model.weights) == len(subclassed_model.weights)
    for a, b in zip(functional_model.weights, subclassed_model.weights):
        np.testing.assert_allclose(a.numpy(), b.numpy())    

    # Because stateless layers do not change the order or number of weights, models can 
    # have compatible architectures even if there are extra/missing stateless layers.
    inputs = Input(shape=(784,), name="digits")
    x = Dense(64, activation="relu", name="dense_1")(inputs)
    x = Dense(64, activation="relu", name="dense_2")(x)
    outputs = Dense(10, name="predictions")(x)
    functional_model = Model(inputs=inputs, outputs=outputs, name="3_layer_mlp")

    inputs = Input(shape=(784,), name="digits")
    x = Dense(64, activation="relu", name="dense_1")(inputs)
    x = Dense(64, activation="relu", name="dense_2")(x)

    # Add a dropout layer, which does not contain any weights.
    x = Dropout(0.5)(x)
    outputs = Dense(10, name="predictions")(x)
    functional_model_with_dropout = Model(
        inputs=inputs, outputs=outputs, name="3_layer_mlp"
    )

    functional_model_with_dropout.set_weights(functional_model.get_weights())


args.step = auto_increment(args.step, args.all)
### Step #11 - Saving & loading only the model's weights values: to/from disk with TF Checkpoint format
if args.step == 11:
    print("\n### Step #11 - Saving & loading only the model's weights values: to/from disk with TF Checkpoint format")

    __doc__='''
    Weights can be saved to disk by calling model.save_weights in the following
    formats:
    - TensorFlow Checkpoint
    - HDF5

    The default format for model.save_weights is TensorFlow checkpoint. There
    are two ways to specify the save format:
    - save_format argument: Set the value to save_format="tf" or
      save_format="h5".
    - path argument: If the path ends with .h5 or .hdf5, then the HDF5 format
      is used.  Other suffixes will result in a TensorFlow checkpoint unless
      save_format is set.
    '''
    print(__doc__)

    # Runnable example
    sequential_model = Sequential([
        InputLayer(input_shape=(784,), name="digits"),
        Dense(64, activation="relu", name="dense_1"),
        Dense(64, activation="relu", name="dense_2"),
        Dense(10, name="predictions"),
    ])

    sequential_model.save_weights("tmp/tf2_g0205/weights")
    sequential_model.load_weights("tmp/tf2_g0205/weights")

    # `assert_consumed` can be used as validation that all variable values have been
    # restored from the checkpoint. See `tf.train.Checkpoint.restore` for other
    # methods in the Status object.
    # load_status.assert_consumed()

    sequential_model.save_weights("tmp/tf2_g0205/weights.h5")
    sequential_model.load_weights("tmp/tf2_g0205/weights.h5")


### End of File
print()
if args.plot:
    plt.show()
debug()

