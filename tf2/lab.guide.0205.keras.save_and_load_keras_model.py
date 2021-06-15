#!/usr/bin/env python

import sys
sys.path.append('./')
sys.path.append('../')

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
from tensorflow.keras.layers import Layer, Dense, InputLayer


### TOC
if args.step == 0:
    toc(__file__)


if True:
    def get_model():
        # Create a simple model.
        inputs = Input(shape=(32,))
        outputs = Dense(1)(inputs)
        model = Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model


args.step = auto_increment(args.step, args.all)
### Step #1 - Whole-model saving & loading: SavedModel format
if args.step == 1:
    print("\n### Step #1 - Whole-model saving & loading: SavedModel format")

    model = get_model()
    
    # Train the model.
    test_input = np.random.random((128, 32))
    test_target = np.random.random((128, 1))
    model.fit(test_input, test_target, verbose=2)

    # Calling `save('my_model')` creates a SavedModel folder `my_model`.
    model.save("tmp/my_model")

    # It can be used to reconstruct the model identically.
    reconstructed_model = tf.keras.models.load_model("tmp/my_model")

    # Let's check:
    np.testing.assert_allclose(
        model.predict(test_input), reconstructed_model.predict(test_input)
    )

    # The reconstructed model is already compiled and has retained the optimizer
    # state, so training can resume:
    reconstructed_model.fit(test_input, test_target, verbose=2)


args.step = auto_increment(args.step, args.all)
### Step #2 - How SavedModel handles custom objects
if args.step == 2:
    print("\n### Step #2 - How SavedModel handles custom objects")

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
    model.save("tmp/my_model")

    # Option 1: Load with the custom_object argument.
    loaded_1 = tf.keras.models.load_model(
        "tmp/my_model", custom_objects={"CustomModel": CustomModel}
    )

    # Option 2: Load without the CustomModel class.

    # Delete the custom-defined model class to ensure that the loader does not have access to it.
    del CustomModel

    loaded_2 = tf.keras.models.load_model("tmp/my_model")
    np.testing.assert_allclose(loaded_1(input_arr), outputs)
    np.testing.assert_allclose(loaded_2(input_arr), outputs)

    logger.info("Original model: {}".format(model))
    logger.info("Model loaded with custom objects: {}".format(loaded_1))
    logger.info("Model loaded without the custom object class: {}".format(loaded_2))


args.step = auto_increment(args.step, args.all)
### Step #3 - Whole-model saving & loading: Keras H5 format
if args.step == 3:
    print("\n### Step #3 - Whole-model saving & loading: Keras H5 format")

    model = get_model()

    # Train the model.
    test_input = np.random.random((128, 32))
    test_target = np.random.random((128, 1))
    model.fit(test_input, test_target)

    # Calling `save('my_model.h5')` creates a h5 file `my_model.h5`.
    model.save("tmp/my_h5_model.h5")

    # It can be used to reconstruct the model identically.
    reconstructed_model = tf.keras.models.load_model("tmp/my_h5_model.h5")

        # Let's check:
    np.testing.assert_allclose(
        model.predict(test_input), reconstructed_model.predict(test_input)
    )

    # The reconstructed model is already compiled and has retained the optimizer
    # state, so training can resume:
    reconstructed_model.fit(test_input, test_target, verbose=2)


args.step = auto_increment(args.step, args.all)
### Step #4 - Saving the architecture: Configuration of a Sequential model or Functional API model
if args.step == 4:
    print("\n### Step #4 - Saving the architecture: Configuration of a Sequential model or Functional API model")
    
    # Note this only applies to models defined using 
    # the functional or Sequential apis not subclassed models
    # Layer example
    layer = Dense(3, activation="relu")
    layer_config = layer.get_config()
    new_layer = Dense.from_config(layer_config)

    # Sequential model example
    model = Sequential([Input((32,)), Dense(1)])
    config = model.get_config()
    new_model = Sequential.from_config(config)

    # Functional model example
    inputs = Input((32,))
    outputs = Dense(1)(inputs)
    model = Model(inputs, outputs)
    config = model.get_config()
    new_model = Model.from_config(config)

    # json
    model = Sequential([Input((32,)), Dense(1)])
    json_config = model.to_json()
    new_model = tf.keras.models.model_from_json(json_config)


args.step = auto_increment(args.step, args.all)
### Step #5 - Saving the architecture: Custom objects 
if args.step == 5:
    print("\n### Step #5 - Saving the architecture: Custom objects")
    pass


args.step = auto_increment(args.step, args.all)
### Step #6 - Saving the architecture: In-memory model cloning
if args.step == 6:
    print("\n### Step #6 - Saving the architecture: In-memory model cloning")
    pass


args.step = auto_increment(args.step, args.all)
### Step #7 - Saving & loading only the model's weights values 
if args.step == 7:
    print("\n### Step #7 - Saving & loading only the model's weights values")

    def create_layer():
        layer = Dense(64, activation="relu", name="dense_2")
        layer.build((None, 784))
        return layer

    layer_1 = create_layer()
    layer_2 = create_layer()

    # Copy weights from layer 1 to layer 2
    layer_2.set_weights(layer_1.get_weights()) 
    

args.step = auto_increment(args.step, args.all)
### Step #8 - Saving & loading only the model's weights values: to/from disk with TF Checkpoint format
if args.step == 8:
    print("\n### Step #8 - Saving & loading only the model's weights values: to/from disk with TF Checkpoint format")

    # Runnable example
    sequential_model = Sequential([
        InputLayer(input_shape=(784,), name="digits"),
        Dense(64, activation="relu", name="dense_1"),
        Dense(64, activation="relu", name="dense_2"),
        Dense(10, name="predictions"),
    ])

    sequential_model.save_weights("tmp/ckpt")
    # load_status = sequential_model.load_weights("tmp/ckpt")

    # `assert_consumed` can be used as validation that all variable values have been
    # restored from the checkpoint. See `tf.train.Checkpoint.restore` for other
    # methods in the Status object.
    # load_status.assert_consumed()


args.step = auto_increment(args.step, args.all)
### Step #9 - Saving & loading only the model's weights values: to/from disk with HDF5 format
if args.step == 9:
    print("\n### Step #9 - Saving & loading only the model's weights values: to/from disk with HDF5 format")

    # Runnable example
    sequential_model = Sequential(
        [
            InputLayer(input_shape=(784,), name="digits"),
            Dense(64, activation="relu", name="dense_1"),
            Dense(64, activation="relu", name="dense_2"),
            Dense(10, name="predictions"),
        ]
    )
    sequential_model.save_weights("tmp/weights.h5")
    # sequential_model.load_weights("tmp/weights.h5")


### End of File
if args.plot:
    plt.show()
debug()

