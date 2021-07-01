#!/usr/bin/env python

import sys
sys.path.append('./')
sys.path.append('../../')

from lab_utils import (
    tf, os, np, plt, logger, ap, BooleanAction,
    debug, toc, auto_increment
)

ap.add_argument('--epochs', type=int, default=3, help='number of epochs: 3*')
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

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Layer, Dense


### TOC
if args.step == 0:
    toc(__file__)


args.step = auto_increment(args.step, args.all)
### Step #1 - The Layer class: the combination of state (weights) and some computation
if args.step == 1:
    print("\n### Step #1 - The Layer class: the combination of state (weights) and some computation")

    __doc__='''
    One of the central abstraction in Keras is the Layer class. A layer
    encapsulates both a state (the layer's "weights") and a transformation from
    inputs to outputs (a "call", the layer's forward pass).
    '''
    print(__doc__)

    class Linear(Layer):
        def __init__(self, units=32, input_dim=32):
            super(Linear, self).__init__()
            # w_init = tf.random_normal_initializer()
            # self.w = tf.Variable(
            #     initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            #     trainable=True,
            # )
            # or
            self.w = self.add_weight(
                shape=(input_dim, units), 
                initializer="random_normal", 
                trainable=True
            )

            # b_init = tf.zeros_initializer()
            # self.b = tf.Variable(
            #     initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
            # )
            # or
            self.b = self.add_weight(
                shape=(units,), 
                initializer="zeros", 
                trainable=True
            ) 

        def call(self, inputs):
            return tf.matmul(inputs, self.w) + self.b

    x = tf.ones((4, 3))
    linear_layer = Linear(10, 3)
    y = linear_layer(x)
    logger.info(f'y is:\n{y}\n')

    assert linear_layer.weights == [linear_layer.w, linear_layer.b]
    logger.info(f"len(weights): {len(linear_layer.weights)}")
    for i, weight in enumerate(linear_layer.weights):
        logger.info(f"weight[{i}]'s shape: {weight.shape}")


args.step = auto_increment(args.step, args.all)
### Step #2 - Layers can have non-trainable weights 
if args.step == 2:
    print("\n### Step #2 - Layers can have non-trainable weights")

    __doc__='''
    Besides trainable weights, you can add non-trainable weights to a layer as
    well. Such weights are meant not to be taken into account during
    backpropagation, when you are training the layer.
    '''
    print(__doc__)

    class ComputeSum(Layer):
        def __init__(self, input_dim):
            super(ComputeSum, self).__init__()
            self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable=False)

        def call(self, inputs):
            self.total.assign_add(tf.reduce_sum(inputs, axis=0))
            return self.total

    # x = tf.ones((2, 2))
    x = tf.constant([[1,2],[3,4]], dtype=tf.float32)
    my_sum = ComputeSum(2)

    y = my_sum(x)
    logger.info(f'y is {y.numpy()}')

    y = my_sum(x)
    logger.info(f'y is {y.numpy()}')

    logger.info(f"len(weights): {len(my_sum.weights)}")
    logger.info(f"len(non-trainable weights): {len(my_sum.non_trainable_weights)}")
    
    # It's not included in the trainable weights:
    logger.info(f"len(trainable_weights): {len(my_sum.trainable_weights)}")


args.step = auto_increment(args.step, args.all)
### Step #3 -- Best practice: deferring weight creation until the shape of the inputs is known
if args.step == 3:
    print("\n### Step #3 -- Best practice: deferring weight creation until the shape of the inputs is known")

    __doc__='''
    In many cases, you may not know in advance the size of your inputs, and you
    would like to lazily create weights when that value becomes known, some
    time after instantiating the layer.

    In the Keras API, we recommend creating layer weights in the build(self,
    inputs_shape) method of your layer. 

    The layer's weights are created dynamically the first time the layer is
    called. (i.e., The call() method of your layer will automatically run build
    the first time it is called) 
    '''
    print(__doc__)

    class Linear(Layer):
        def __init__(self, units=32):
            super(Linear, self).__init__()
            self.units = units

        def build(self, input_shape):
            self.w = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer="random_normal",
                trainable=True,
            )
            self.b = self.add_weight(
                shape=(self.units,), 
                initializer="random_normal", 
                trainable=True
            )

        def call(self, inputs):
            return tf.matmul(inputs, self.w) + self.b    

    # At instantiation, we don't know on what inputs this is going to get called
    linear_layer = Linear(32)

    x = tf.ones((3,4))
    y = linear_layer(x)

    logger.info(f"len(weights): {len(linear_layer.weights)}")
    for i, weight in enumerate(linear_layer.weights):
        logger.info(f"weight[{i}]'s shape: {weight.shape}")


args.step = auto_increment(args.step, args.all)
### Step #4 - Layers are recursively composable
if args.step == 4:
    print("\n### Step #4 - Layers are recursively composable")
    
    __doc__='''
    If you assign a Layer instance as an attribute of another Layer, the outer
    layer will start tracking the weights of the inner layer.

    We recommend creating such sublayers in the __init__() method (since the
    sublayers will typically have a build method, they will be built when the
    outer layer gets built).
    '''
    print(__doc__)

    class Linear(Layer):
        def __init__(self, units=32):
            super(Linear, self).__init__()
            self.units = units

        def build(self, input_shape):
            self.w = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer="random_normal",
                trainable=True,
            )
            self.b = self.add_weight(
                shape=(self.units,), 
                initializer="random_normal", 
                trainable=True
            )

        def call(self, inputs):
            return tf.matmul(inputs, self.w) + self.b    

    # Let's assume we are reusing the Linear class
    # with a `build` method that we defined above.
    class MLPBlock(Layer):
        def __init__(self):
            super(MLPBlock, self).__init__()
            self.linear_1 = Linear(32)
            self.linear_2 = Linear(32)
            self.linear_3 = Linear(1)

        def call(self, inputs):
            x = self.linear_1(inputs)
            x = tf.nn.relu(x)
            x = self.linear_2(x)
            x = tf.nn.relu(x)
            return self.linear_3(x)

    mlp = MLPBlock()
    y = mlp(tf.ones(shape=(3, 64)))  # The first call to the `mlp` will create the weights
    logger.info(f"len(trainable weights): {len(mlp.trainable_weights)}")
    logger.info(f"len(weights): {len(mlp.weights)}")
    for i, weight in enumerate(mlp.weights):
        logger.info(f"weight[{i}]'s shape: {weight.shape}")


args.step = auto_increment(args.step, args.all)
### Step #5 - The add_loss() method
if args.step == 5:
    print("\n### Step #5 - The add_loss() method")
    
    __doc__='''
    When writing the call() method of a layer, you can create loss tensors that
    you will want to use later, when writing your training loop. This is doable
    by calling self.add_loss(value).

    These losses (including those created by any inner layer) can be retrieved
    via layer.losses. This property is reset at the start of every __call__()
    to the top-level layer, so that layer.losses always contains the loss
    values created during the last forward pass.

    In addition, the loss property of the outer layer also contains
    regularization losses created for the weights of any inner layer.

    These losses are meant to be taken into account when writing training
    loops.
    '''
    print(__doc__)

    # A layer that creates an activity regularization loss
    class ActivityRegularizationLayer(Layer):
        def __init__(self, rate=1e-2):
            super(ActivityRegularizationLayer, self).__init__()
            self.rate = rate

        def call(self, inputs):
            self.add_loss(self.rate * tf.reduce_sum(inputs))
            return inputs

    class OuterLayer(Layer):
        def __init__(self):
            super(OuterLayer, self).__init__()
            self.activity_reg = ActivityRegularizationLayer(1e-2)

        def call(self, inputs):
            return self.activity_reg(inputs)

    layer = OuterLayer()
    assert len(layer.losses) == 0  # No losses yet since the layer has never been called

    _ = layer(tf.zeros(1, 1))
    assert len(layer.losses) == 1  # We created one loss value

    # `layer.losses` gets reset at the start of each __call__
    _ = layer(tf.zeros(1, 1))
    assert len(layer.losses) == 1  # This is the loss created during the call above

    class OuterLayerWithKernelRegularizer(Layer):
        def __init__(self):
            super(OuterLayerWithKernelRegularizer, self).__init__()
            self.dense = Dense(
                32, kernel_regularizer=tf.keras.regularizers.l2(1e-3)
            )

        def call(self, inputs):
            return self.dense(inputs)

    layer = OuterLayerWithKernelRegularizer()
    _ = layer(tf.zeros((1, 1)))

    # This is `1e-3 * sum(layer.dense.kernel ** 2)`,
    # created by the `kernel_regularizer` above.
    logger.info(f'outer_layer.losses:\n{layer.losses}\n')

    # These losses are meant to be taken into account when writing training loops
    # Instantiate an optimizer.
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Iterate over the batches of a dataset.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    y_train = y_train.astype("float32")

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

    layer = OuterLayerWithKernelRegularizer()

    for x_batch_train, y_batch_train in train_dataset:
        with tf.GradientTape() as tape:
            logits = layer(x_batch_train)  # Logits for this minibatch
            # Loss value for this minibatch
            loss_value = loss_fn(y_batch_train, logits)
            # Add extra losses created during this forward pass:
            loss_value += sum(layer.losses)

        grads = tape.gradient(loss_value, layer.trainable_weights)
        optimizer.apply_gradients(zip(grads, layer.trainable_weights))
    
    # These losses also work seamlessly with fit() (they get automatically summed
    # and added to the main loss, if any):
    inputs = Input(shape=(3,))
    outputs = ActivityRegularizationLayer()(inputs)
    model = Model(inputs, outputs)

    # If there is a loss passed in `compile`, the regularization
    # losses get added to it
    model.compile(optimizer="adam", loss="mse")
    logger.info('layer.losses are added to the main loss:')
    model.fit(
        np.random.random((2, 3)), 
        np.random.random((2, 3)),
        epochs=args.epochs,
        verbose=2
    )
    print()

    # It's also possible not to pass any loss in `compile`,
    # since the model already has a loss to minimize, via the `add_loss`
    # call during the forward pass!
    model.compile(optimizer="adam")
    model.fit(
        np.random.random((2, 3)), np.random.random((2, 3)),
        epochs=args.epochs,
        verbose=2
    )


args.step = auto_increment(args.step, args.all)
### Step #6 - The add_metric() method
if args.step == 6:
    print("\n### Step #6 - The add_metric() method")

    class LogisticEndpoint(Layer):
        def __init__(self, name=None):
            super(LogisticEndpoint, self).__init__(name=name)
            self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            self.accuracy_fn = tf.keras.metrics.BinaryAccuracy()

        def call(self, targets, logits, sample_weights=None):
            # Compute the training-time loss value and add it
            # to the layer using `self.add_loss()`.
            loss = self.loss_fn(targets, logits, sample_weights)
            self.add_loss(loss)

            # Log accuracy as a metric and add it
            # to the layer using `self.add_metric()`.
            acc = self.accuracy_fn(targets, logits, sample_weights)
            self.add_metric(acc, name="accuracy")

            # Return the inference-time prediction tensor (for `.predict()`).
            return tf.nn.softmax(logits)

    layer = LogisticEndpoint()

    targets = tf.ones((2, 2))
    logits = tf.ones((2, 2))
    y = layer(targets, logits)

    logger.info(f"layer.metrics: {layer.metrics}")
    logger.info("current accuracy value: {}\n".format(float(layer.metrics[0].result())))

    # Just like for add_loss(), these metrics are tracked by fit():
    inputs = Input(shape=(3,), name="inputs")
    targets = Input(shape=(10,), name="targets")
    logits = Dense(10)(inputs)
    predictions = LogisticEndpoint(name="predictions")(logits, targets)

    model = Model(inputs=[inputs, targets], outputs=predictions)
    model.compile(optimizer="adam")

    data = {
        "inputs": np.random.random((3, 3)),
        "targets": np.random.random((3, 10)),
    }
    model.fit(
        data,
        epochs=args.epochs, 
        verbose=2
    )


args.step = auto_increment(args.step, args.all)
### Step #7 - You can optionally enable serialization on your layers
if args.step == 7:
    print("\n### Step #7 - You can optionally enable serialization on your layers")

    __doc__='''
    If you need your custom layers to be serializable as part of a Functional
    model, you can optionally implement a get_config() method.
    
    Note that the __init__() method of the base Layer class takes some keyword
    arguments, in particular a name and a dtype. It's good practice to pass
    these arguments to the parent class in __init__() and to include them in
    the layer config.
    '''
    print(__doc__)

    class Linear(Layer):
        def __init__(self, units=32, **kwargs):
            super(Linear, self).__init__(**kwargs)
            self.units = units

        def build(self, input_shape):
            self.w = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer="random_normal",
                trainable=True,
            )
            self.b = self.add_weight(
                shape=(self.units,), initializer="random_normal", trainable=True
            )

        def call(self, inputs):
            return tf.matmul(inputs, self.w) + self.b

        def get_config(self):
            config = super(Linear, self).get_config()
            config.update({"units": self.units})
            return config

    layer = Linear(64)
    config = layer.get_config()
    logger.info(f'layer.get_config():\n{config}')
    new_layer = Linear.from_config(config)


args.step = auto_increment(args.step, args.all)
### Step #8 - Privileged training argument in the call() method
if args.step == 8:
    print("\n### Step #8 - Privileged training argument in the call() method")

    __doc__='''
    Some layers, in particular the BatchNormalization layer and the Dropout
    layer, have different behaviors during training and inference. For such
    layers, it is standard practice to expose a training (boolean) argument in
    the call() method.

    By exposing this argument in call(), you enable the built-in training and
    evaluation loops (e.g. fit()) to correctly use the layer in training and
    inference.
    '''
    print(__doc__)

    class CustomDropout(Layer):
        def __init__(self, rate, **kwargs):
            super(CustomDropout, self).__init__(**kwargs)
            self.rate = rate

        def call(self, inputs, training=None):
            if training:
                return tf.nn.dropout(inputs, rate=self.rate)
            return inputs


args.step = auto_increment(args.step, args.all)
### Step #9 - Privileged mask argument in the call() method
if args.step == 9:
    print("\n### Step #9 - Privileged mask argument in the call() method")
    pass


args.step = auto_increment(args.step, args.all)
### Step #10 - The Model class
if args.step == 10:
    print("\n### Step #10 - The Model class")

    __doc__='''
    In general, you will use the Layer class to define inner computation
    blocks, and will use the Model class to define the outer model -- the
    object you will train.

    For instance, in a ResNet50 model, you would have several ResNet blocks
    subclassing Layer, and a single Model encompassing the entire ResNet50
    network.

    The Model class has the same API as Layer, with the following differences:
    - It exposes built-in training, evaluation, and prediction loops
      (model.fit(), model.evaluate(), model.predict()).
    - It exposes the list of its inner layers, via the model.layers property.
    - It exposes saving and serialization APIs (save(), save_weights()...)

    Effectively, the Layer class corresponds to what we refer to in the
    literature as a "layer" (as in "convolution layer" or "recurrent layer") or
    as a "block" (as in "ResNet block" or "Inception block").

    Meanwhile, the Model class corresponds to what is referred to in the
    literature as a "model" (as in "deep learning model") or as a "network" (as
    in "deep neural network").
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #11 - Putting it all together: an end-to-end example
if args.step == 11: 
    print("\n### Step #11 - Putting it all together: an end-to-end example")

    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype("float32") / 255

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

    class Sampling(Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    class Encoder(Layer):
        """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""
        def __init__(self, latent_dim=32, intermediate_dim=64, name="encoder", **kwargs):
            super(Encoder, self).__init__(name=name, **kwargs)
            self.dense_proj = Dense(intermediate_dim, activation="relu")
            self.dense_mean = Dense(latent_dim)
            self.dense_log_var = Dense(latent_dim)
            self.sampling = Sampling()

        def call(self, inputs):
            x = self.dense_proj(inputs)
            z_mean = self.dense_mean(x)
            z_log_var = self.dense_log_var(x)
            z = self.sampling((z_mean, z_log_var))
            return z_mean, z_log_var, z

    class Decoder(Layer):
        """Converts z, the encoded digit vector, back into a readable digit."""
        def __init__(self, original_dim, intermediate_dim=64, name="decoder", **kwargs):
            super(Decoder, self).__init__(name=name, **kwargs)
            self.dense_proj = Dense(intermediate_dim, activation="relu")
            self.dense_output = Dense(original_dim, activation="sigmoid")

        def call(self, inputs):
            x = self.dense_proj(inputs)
            return self.dense_output(x)

    class VariationalAutoEncoder(Model):
        """Combines the encoder and decoder into an end-to-end model for training."""
        def __init__(self, original_dim, intermediate_dim=64, latent_dim=32, name="autoencoder", **kwargs):
            super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
            self.original_dim = original_dim
            self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
            self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

        def call(self, inputs):
            z_mean, z_log_var, z = self.encoder(inputs)
            reconstructed = self.decoder(z)
            # Add KL divergence regularization loss.
            kl_loss = -0.5 * tf.reduce_mean(
                z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
            )
            self.add_loss(kl_loss)
            return reconstructed

    original_dim = 784
    vae = VariationalAutoEncoder(original_dim, 64, 32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    loss_metric = tf.keras.metrics.Mean()

    epochs = args.epochs

    logger.info('with training loop:')
    # Iterate over epochs.
    for epoch in range(epochs):
        logger.info("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, x_batch_train in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                reconstructed = vae(x_batch_train)
                # Compute reconstruction loss
                loss = mse_loss_fn(x_batch_train, reconstructed)
                loss += sum(vae.losses)  # Add KLD regularization loss

            grads = tape.gradient(loss, vae.trainable_weights)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))

            # loss_metric(loss)
            loss_metric.update_state(loss)

            if step % 100 == 0:
                logger.info("step %d: mean loss = %.4f" % (step, loss_metric.result()))
        print()

    logger.info('against fit():')
    vae = VariationalAutoEncoder(784, 64, 32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
    vae.fit(
        x_train, x_train, 
        epochs=args.epochs, 
        batch_size=64, 
        verbose=2
    )


args.step = auto_increment(args.step, args.all)
### Step #12 - Beyond object-oriented development: the Functional API
if args.step == 12:
    print("\n### Step #12 - Beyond object-oriented development: the Functional API")

    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype("float32") / 255

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

    class Sampling(Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    original_dim = 784
    intermediate_dim = 64
    latent_dim = 32

    # Define encoder model.
    original_inputs = Input(shape=(original_dim,), name="encoder_input")
    x = Dense(intermediate_dim, activation="relu")(original_inputs)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()((z_mean, z_log_var))
    encoder = Model(inputs=original_inputs, outputs=z, name="encoder")

    # Define decoder model.
    latent_inputs = Input(shape=(latent_dim,), name="z_sampling")
    x = Dense(intermediate_dim, activation="relu")(latent_inputs)
    outputs = Dense(original_dim, activation="sigmoid")(x)
    decoder = Model(inputs=latent_inputs, outputs=outputs, name="decoder")

    # Define VAE model.
    outputs = decoder(z)
    vae = Model(inputs=original_inputs, outputs=outputs, name="vae")

    # Add KL divergence regularization loss.
    kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    vae.add_loss(kl_loss)

    # Train.
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
    vae.fit(
        x_train, x_train, 
        epochs=args.epochs, 
        batch_size=64, 
        verbose=2
    )


### End of File
print()
if args.plot:
    plt.show()
debug()

