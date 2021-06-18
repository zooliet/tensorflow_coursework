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
### Step #1 - Setup and basic usage
if args.step == 1:
    print("\n### Step #1 - Setup and basic usage")

    # In Tensorflow 2.0, eager execution is enabled by default.
    logger.info(f'tf.executing_eagerly(): {tf.executing_eagerly()}')

    # run TensorFlow operations and the results will return immediately
    x = [[2.]] # 1x1
    m = tf.matmul(x, x) 
    # m is a tf.Tensor object references the concrete value instead of symbolic handles
    # to nodes in a computational graph
    logger.info("hello, {}".format(m))

    logger.info('Eager execution works nicely with NumPy:')
    a = tf.constant([[1, 2], [3, 4]]) # 2x2
    print(a, '\n') # a is tf.Tensor

    logger.info('Broadcasting support:')
    b = tf.add(a, 1)
    print(b, '\n')

    logger.info('Operator overloading is supported:')
    print(a * b, '\n')

    logger.info('Use NumPy values:')
    c = np.multiply(a, b) # a,b are tf.Tensor, c is of an numpy
    print(c, type(c), '\n')

    logger.info('Obtain numpy value from a tensor:')
    print(a.numpy(), '\n')


args.step = auto_increment(args.step, args.all)
### Step #2 - Dynamic control flow
if args.step == 2:
    print("\n### Step #2 - Dynamic control flow")

    # A major benefit of eager execution is that all the functionality 
    # of the host language is available while your model is executing. 
    # This has conditionals that depend on tensor values and it prints 
    # these values at runtime
    def fizzbuzz(max_num):
        counter = tf.constant(0)
        max_num = tf.convert_to_tensor(max_num) # == tf.constant(max_num)
        for num in range(1, max_num.numpy()+1):
            num = tf.constant(num)
            if int(num % 3) == 0 and int(num % 5) == 0:
                print('FizzBuzz')
            elif int(num % 3) == 0:
                print('Fizz')
            elif int(num % 5) == 0:
                print('Buzz')
            else:
                print(num.numpy())
            counter += 1

    fizzbuzz(15)


args.step = auto_increment(args.step, args.all)
### Step #3 - Eager training: Computing gradients
if args.step == 3:
    print("\n### Step #3 - Eager training: Computing gradients")

    w = tf.Variable([[1.0]])
    with tf.GradientTape() as tape:
        loss = w * w

    grad = tape.gradient(loss, w)
    print(grad)  # => tf.Tensor([[ 2.]], shape=(1, 1), dtype=float32)


args.step = auto_increment(args.step, args.all)
### Step #4 - Eager training: Train a model
if args.step == 4:
    print("\n### Step #4 - Eager training: Train a model")

    # Fetch and format the mnist data
    (mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32),
         tf.cast(mnist_labels,tf.int64))
    )
    dataset = dataset.shuffle(1000).batch(32)

    # Build the model
    mnist_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16,[3,3], activation='relu', input_shape=(None, None, 1)),
        tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10)
    ])

    logger.info('Even without training, call the model and inspect the output in eager execution:')
    for images, labels in dataset.take(1):
        print("Logits: ", mnist_model(images[0:1]).numpy(), '\n')

    logger.info('training loop implemented with eager:')
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    loss_history = []
    # @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            logits = mnist_model(images, training=True)

            # Add asserts to check the shape of the output.
            tf.debugging.assert_equal(logits.shape, (32, 10))
            loss_value = loss_object(labels, logits)

        loss_history.append(loss_value.numpy().mean())
        grads = tape.gradient(loss_value, mnist_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))

    def train(epochs):
        for epoch in range(epochs):
            for (batch, (images, labels)) in enumerate(dataset):
                train_step(images, labels)
            print ('Epoch {} finished'.format(epoch))

    train(epochs = 3)
    print('')

    if args.plot:
        plt.plot(loss_history)
        plt.xlabel('Batch #')
        plt.ylabel('Loss [entropy]')
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #5 - Eager training: Variables and optimizers
if args.step == 5:
    print("\n### Step #5 - Eager training: Variables and optimizers")

    class Linear(tf.keras.Model):
        def __init__(self):
            super(Linear, self).__init__()
            self.W = tf.Variable(5., name='weight')
            self.B = tf.Variable(10., name='bias')
        def call(self, inputs):
            return inputs * self.W + self.B

    # A toy dataset of points around 3 * x + 2
    NUM_EXAMPLES = 2000
    training_inputs = tf.random.normal([NUM_EXAMPLES])
    noise = tf.random.normal([NUM_EXAMPLES])
    training_outputs = training_inputs * 3 + 2 + noise

    # The loss function to be optimized
    def loss(model, inputs, targets):
        error = model(inputs) - targets
        return tf.reduce_mean(tf.square(error))

    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets)
        return tape.gradient(loss_value, [model.W, model.B])

    model = Linear()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))

    steps = 300
    for i in range(steps):
        grads = grad(model, training_inputs, training_outputs)
        optimizer.apply_gradients(zip(grads, [model.W, model.B]))
        if i % 20 == 0:
            print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))

    print("Final loss: {:.3f}\n".format(loss(model, training_inputs, training_outputs)))
    print("W = {}, B = {}\n".format(model.W.numpy(), model.B.numpy()))


args.step = auto_increment(args.step, args.all)
### Step #6 - Eager training: Object-based saving
if args.step == 6:
    print("\n### Step #6 - Eager training: Object-based saving")

    x = tf.Variable(10.)
    checkpoint = tf.train.Checkpoint(x=x)
    x.assign(2.)   # Assign a new value to the variables and save.
    checkpoint_path = 'tmp/ckpt/'
    checkpoint.save(checkpoint_path)

    x.assign(11.)  # Change the variable after saving.

    # Restore values from the checkpoint
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
    print(x, '\n')  # => 2.0

    # To save and load models, tf.train.Checkpoint stores the internal state of objects, 
    # without requiring hidden variables. To record the state of a model, an optimizer, 
    # and a global step, pass them to a tf.train.Checkpoint:
    model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
      tf.keras.layers.GlobalAveragePooling2D(),
      tf.keras.layers.Dense(10)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    checkpoint_dir = 'tmp/model_path'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    root = tf.train.Checkpoint(optimizer=optimizer, model=model)
    root.save(checkpoint_prefix)
    root.restore(tf.train.latest_checkpoint(checkpoint_dir))


args.step = auto_increment(args.step, args.all)
### Step #7 - Eager training: Object-oriented metrics
if args.step == 7:
    print("\n### Step #7 - Eager training: Object-oriented metrics")

    m = tf.keras.metrics.Mean("loss")
    m(0)
    m(5)
    logger.info(f'm.result(): {m.result()}')

    m([8, 9])
    logger.info(f'm.result(): {m.result()}')


args.step = auto_increment(args.step, args.all)
### Step #8 - Eager training: Summaries and TensorBoard
if args.step == 8:
    print("\n### Step #8 - Eager training: Summaries and TensorBoard")

    # TensorBoard is a visualization tool for understanding, debugging and optimizing 
    # the model training process. 
    # It uses summary events that are written while executing the program.
    # You can use tf.summary to record summaries of variable in eager execution. 
    # For example, to record summaries of loss once every 100 training steps:
    logdir = "tmp/tb/"
    writer = tf.summary.create_file_writer(logdir)

    steps = 1000
    with writer.as_default():  # or call writer.set_as_default() before the loop.
        for i in range(steps):
            step = i + 1
            # Calculate loss with your real train function.
            loss = 1 - 0.001 * step
            if step % 100 == 0:
                tf.summary.scalar('loss', loss, step=step)

    print(*os.listdir(logdir), sep='\n')
    print('')


args.step = auto_increment(args.step, args.all)
### Step #9 - Advanced automatic differentiation topics: Dynamic models
if args.step == 9:
    print("\n### Step #9 - Advanced automatic differentiation topics: Dynamic models")

    def line_search_step(fn, init_x, rate=1.0):
        with tf.GradientTape() as tape:
            # Variables are automatically tracked.
            # But to calculate a gradient from a tensor, you must `watch` it.
            tape.watch(init_x)
            value = fn(init_x)

        grad = tape.gradient(value, init_x)
        grad_norm = tf.reduce_sum(grad * grad)
        init_value = value
        while value > init_value - rate * grad_norm:
            x = init_x - rate * grad
            value = fn(x)
            rate /= 2.0

        return x, value


args.step = auto_increment(args.step, args.all)
### Step #10 - Advanced automatic differentiation topics: Custom gradients
if args.step == 10:
    print("\n### Step #10 - Advanced automatic differentiation topics: Custom gradients")

    @tf.custom_gradient
    def clip_gradient_by_norm(x, norm):
        y = tf.identity(x)
        def grad_fn(dresult):
            return [tf.clip_by_norm(dresult, norm), None]
        return y, grad_fn

    def log1pexp(x):
        return tf.math.log(1 + tf.exp(x))

    def grad_log1pexp(x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            value = log1pexp(x)
        return tape.gradient(value, x)

    # The gradient computation works fine at x = 0.
    grad_log1pexp(tf.constant(0.)).numpy()

    # However, x = 100 fails because of numerical instability.
    grad_log1pexp(tf.constant(100.)).numpy()

    @tf.custom_gradient
    def log1pexp(x):
        e = tf.exp(x)
        def grad(dy):
            return dy * (1 - 1 / (1 + e))
        return tf.math.log(1 + e), grad

    def grad_log1pexp(x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            value = log1pexp(x)
        return tape.gradient(value, x)

    # As before, the gradient computation works fine at x = 0.
    grad_log1pexp(tf.constant(0.)).numpy()

    # And the gradient computation also works at x = 100.
    grad_log1pexp(tf.constant(100.)).numpy()


args.step = auto_increment(args.step, args.all)
### Step #11 - Performance
if args.step == 11:
    print("\n### Step #11 - Performance")

    def measure(x, steps):
        # TensorFlow initializes a GPU the first time it's used, exclude from timing.
        tf.matmul(x, x)
        start = time.time()
        for i in range(steps):
            x = tf.matmul(x, x)
        # tf.matmul can return before completing the matrix multiplication
        # (e.g., can return after enqueing the operation on a CUDA stream).
        # The x.numpy() call below will ensure that all enqueued operations
        # have completed (and will also copy the result to host memory,
        # so we're including a little more than just the matmul operation
        # time).
        _ = x.numpy()
        end = time.time()
        return end - start

    shape = (1000, 1000)
    steps = 200
    logger.info("Time to multiply a {} matrix by itself {} times:".format(shape, steps))

    # Run on CPU:
    with tf.device("/cpu:0"):
        logger.info("CPU: {} secs".format(measure(tf.random.normal(shape), steps)))

    # Run on GPU, if available:
    if tf.config.list_physical_devices("GPU"):
        with tf.device("/gpu:0"):
            logger.info("GPU: {} secs\n".format(measure(tf.random.normal(shape), steps)))
    else:
        logger.info("GPU: not found\n")


args.step = auto_increment(args.step, args.all)
### Step #12 - Performance: Benchmarks
if args.step == 12:
    print("\n### Step #12 - Performance: Benchmarks")
    
    str = '''
    For compute-heavy models, such as ResNet50 training on a GPU, eager execution 
    performance is comparable to tf.function execution. But this gap grows larger for 
    models with less computation and there is work to be done for optimizing hot code paths 
    for models with lots of small operations.
    '''
    print(str)


args.step = auto_increment(args.step, args.all)
### Step #13 - Work with functions
if args.step == 13:
    print("\n### Step #13 - Work with functions")

    str = '''
    While eager execution makes development and debugging more interactive, TensorFlow 1.x
    style graph execution has advantages for distributed training, performance optimizations, 
    and production deployment. To bridge this gap, TensorFlow 2.0 introduces functions via 
    the tf.function API. For more information, see the tf.function guide.
    '''
    print(str)


### End of File
if args.plot:
    plt.show()
debug()
