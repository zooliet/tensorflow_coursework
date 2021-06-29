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
### Step #1 - Computing gradients
if args.step == 1:
    print("\n### Step #1 - Computing gradients")

    __doc__='''
    Automatic differentiation is useful for implementing machine learning
    algorithms such as backpropagation for training neural networks.

    To differentiate automatically, TensorFlow needs to remember what
    operations happen in what order during the forward pass. Then, during the
    backward pass, TensorFlow traverses this list of operations in reverse
    order to compute gradients.
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #2 - Gradient tapes
if args.step == 2:
    print("\n### Step #2 - Gradient tapes")

    __doc__='''
    TensorFlow provides the tf.GradientTape API for automatic differentiation;
    that is, computing the gradient of a computation with respect to some
    inputs, usually tf.Variables. TensorFlow "records" relevant operations
    executed inside the context of a tf.GradientTape onto a "tape". TensorFlow
    then uses that tape to compute the gradients of a "recorded" computation
    using reverse mode differentiation.
    '''
    print(__doc__)

    x = tf.Variable(3.0)
    with tf.GradientTape() as tape:
        y = x**2

    __doc__='''
    Once you've recorded some operations, use GradientTape.gradient(target,
    sources) to calculate the gradient of some target (often a loss) relative
    to some source (often the model's variables):
    '''
    print(__doc__)

    # dy = 2x * dx
    dy_dx = tape.gradient(y, x)
    logger.info(f'dy_dx=tape.gradient(y,x): {dy_dx.numpy()}')

    logger.info('tf.GradientTape works as easily on any tensor:')
    w = tf.Variable(tf.random.normal((3, 2)), name='w')
    b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
    x = [[1., 2., 3.]]

    with tf.GradientTape(persistent=True) as tape:
        y = x @ w + b
        loss = tf.reduce_mean(y**2)

    [dl_dw, dl_db] = tape.gradient(loss, [w, b])
    print('d(loss)/dw:\n', dl_dw, '\n')
    print('d(loss)/db:\n', dl_db, '\n')
    
    logger.info('passing a dictionary of variables for gradient calculation:')
    my_vars = {
        'w': w,
        'b': b
    }

    grad = tape.gradient(loss, my_vars)
    print(grad['w'], '\n')
    print(grad['b'], '\n')


args.step = auto_increment(args.step, args.all)
### Step #3 - Gradients with respect to a model
if args.step == 3:
    print("\n### Step #3 - Gradients with respect to a model")

    __doc__='''
    It's common to collect tf.Variables into a tf.Module or one of its
    subclasses (layers.Layer, keras.Model) for checkpointing and exporting.

    In most cases, you will want to calculate gradients with respect to a
    model's trainable variables. Since all subclasses of tf.Module aggregate
    their variables in the Module.trainable_variables property, you can
    calculate these gradients in a few lines of code
    '''
    print(__doc__)

    layer = tf.keras.layers.Dense(2, activation='relu')
    x = tf.constant([[1., 2., 3.]])

    with tf.GradientTape() as tape:
        # Forward pass
        y = layer(x)
        loss = tf.reduce_mean(y**2)

    # Calculate gradients with respect to every trainable variable
    grad = tape.gradient(loss, layer.trainable_variables)

    for var, g in zip(layer.trainable_variables, grad):
        print(f'{var.name}, shape: {g.shape}')
    print('')


args.step = auto_increment(args.step, args.all)
### Step #4 - Controlling what the tape watches
if args.step == 4:
    print("\n### Step #4 - Controlling what the tape watches")
    
    __doc__='''
    The default behavior is to record all operations after accessing a
    trainable tf.Variable. The reasons for this are:
    - The tape needs to know which operations to record in the forward pass to
      calculate the gradients in the backwards pass.
    - The tape holds references to intermediate outputs, so you don't want to
      record unnecessary operations.
    - The most common use case involves calculating the gradient of a loss with
      respect to all a model's trainable variables.
    '''
    print(__doc__)

    logger.info('Not all variables are watched:')
    # A trainable variable
    x0 = tf.Variable(3.0, name='x0')
    # Not trainable
    x1 = tf.Variable(3.0, name='x1', trainable=False)
    # Not a Variable: A variable + tensor returns a tensor.
    x2 = tf.Variable(2.0, name='x2') + 1.0
    # Not a variable
    x3 = tf.constant(3.0, name='x3')

    with tf.GradientTape() as tape:
        y = (x0**2) + (x1**2) + (x2**2)

    grad = tape.gradient(y, [x0, x1, x2, x3])

    for g in grad:
        print(g)
    print('')

    logger.info('list the variables being watched by the tape:')
    print([var.name for var in tape.watched_variables()], '\n')

    logger.info('To record gradients with respect to a tf.Tensor, use GradientTape.watch(x):')
    x = tf.constant(3.0)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = x**2

    # dy = 2x * dx
    dy_dx = tape.gradient(y, x)
    print(dy_dx.numpy(), '\n')


args.step = auto_increment(args.step, args.all)
### Step #5 - Intermediate results
if args.step == 5:
    print("\n### Step #5 - Intermediate results")

    x = tf.constant(3.0)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = x * x
        z = y * y

    # Use the tape to compute the gradient of z with respect to the
    # intermediate value y.
    # dz_dy = 2 * y and y = x ** 2 = 9
    print('tape.gradient(z,y): ', tape.gradient(z, y).numpy())

    __doc__='''
    By default, the resources held by a GradientTape are released as soon as
    the GradientTape.gradient method is called. To compute multiple gradients
    over the same computation, create a gradient tape with persistent=True.
    This allows multiple calls to the gradient method as resources are released
    when the tape object is garbage collected.:
    '''
    print(__doc__)

    x = tf.constant([1, 3.0])
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = x * x
        z = y * y

    print('tape.gradient(z,x):\n', tape.gradient(z, x).numpy(), '\n')  # 108.0 (4 * x**3 at x = 3)
    print('tape.gradient(z,y):\n', tape.gradient(y, x).numpy(), '\n')  # 6.0 (2 * x)

    # Drop the reference to the tape
    del tape  


args.step = auto_increment(args.step, args.all)
### Step #6 - Notes on performance
if args.step == 6:
    print("\n### Step #6 - Notes on performance")

    __doc__='''
    There is a tiny overhead associated with doing operations inside a gradient
    tape context. For most eager execution this will not be a noticeable cost,
    but you should still use tape context around the areas only where it is
    required.

    Gradient tapes use memory to store intermediate results, including inputs
    and outputs, for use during the backwards pass.

    For efficiency, some ops (like ReLU) don't need to keep their intermediate
    results and they are pruned during the forward pass. However, if you use
    persistent=True on your tape, nothing is discarded and your peak memory
    usage will be higher.
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #7 - Gradients of non-scalar targets
if args.step == 7:
    print("\n### Step #7 - Gradients of non-scalar targets")

    logger.info('A gradient is fundamentally an operation on a scalar:')
    x = tf.Variable(2.0)
    with tf.GradientTape(persistent=True) as tape:
        y0 = x**2
        y1 = 1 / x

    print('tape.gradient(y0,x): ', tape.gradient(y0, x).numpy())
    print('tape.gradient(y1,x): ', tape.gradient(y1, x).numpy(), '\n')


    __doc__='''
    If you ask for the gradient of multiple targets, the result for each source
    is:
    - The gradient of the sum of the targets, or equivalently
    - The sum of the gradients of each target.
    '''
    print(__doc__)

    x = tf.Variable(2.0)
    with tf.GradientTape() as tape:
        y0 = x**2
        y1 = 1 / x

    print(
        "tape.gradient({'y0': y0, 'y1': y1}, x): ", 
        tape.gradient({'y0': y0, 'y1': y1}, x).numpy(), '\n'
    )

    x = tf.Variable(2.)
    with tf.GradientTape() as tape:
        y = x * [3., 4.]

    print(
        "tape.gradient(y, x): ", 
        tape.gradient(y, x).numpy(), '\n'
    )

    x = tf.linspace(-10.0, 10.0, 200+1)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = tf.nn.sigmoid(x)

    dy_dx = tape.gradient(y, x)

    if args.plot:
        plt.figure()
        plt.plot(x, y, label='y')
        plt.plot(x, dy_dx, label='dy/dx')
        plt.legend()
        _ = plt.xlabel('x')
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #8 - Control flow
if args.step == 8:
    print("\n### Step #8 - Control flow")

    x = tf.constant(1.0)
    v0 = tf.Variable(2.0)
    v1 = tf.Variable(2.0)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        if x > 0.0:
            result = v0
        else:
            result = v1**2 

    dv0, dv1 = tape.gradient(result, [v0, v1])

    print('tape.gradient(result, v0): ', dv0)
    print('tape.gradient(result, v1): ', dv1, '\n')  


args.step = auto_increment(args.step, args.all)
### Step #9 - Getting a gradient of None
if args.step == 9:
    print("\n### Step #9 - Getting a gradient of None")

    x = tf.Variable(2.)
    y = tf.Variable(3.)
    with tf.GradientTape() as tape:
        z = y * y

    logger.info('When a target is not connected to a source you will get a gradient of None:')
    print('tape.gradient(z,x): ', tape.gradient(z, x), '\n')


args.step = auto_increment(args.step, args.all)
### Step #10 - Getting a gradient of None: Replaced a variable with a tensor
if args.step == 10:
    print("\n### Step #10 - Getting a gradient of None: Replaced a variable with a tensor")

    x = tf.Variable(2.0)
    for epoch in range(2):
        with tf.GradientTape() as tape:
            y = x+1

        print(type(x).__name__, ":", tape.gradient(y, x))
        x = x + 1   # This should be `x.assign_add(1)`
    print('')


args.step = auto_increment(args.step, args.all)
### Step #11 - Getting a gradient of None: Did calculations outside of TensorFlow
if args.step == 11:
    print("\n### Step #11 - Getting a gradient of None: Did calculations outside of TensorFlow")

    x = tf.Variable([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    with tf.GradientTape() as tape:
        x2 = x**2

        # This step is calculated with NumPy
        y = np.mean(x2, axis=0)

        # Like most ops, reduce_mean will cast the NumPy array to a constant tensor
        # using `tf.convert_to_tensor`.
        y = tf.reduce_mean(y, axis=0)

    print('tape.gradient(y,x): ', tape.gradient(y, x), '\n')


args.step = auto_increment(args.step, args.all)
### Step #12 - Getting a gradient of None: Took gradients through an integer or string
if args.step == 12:
    print("\n### Step #12 - Getting a gradient of None: Took gradients through an integer or string")

    x = tf.constant(10) # should be 10.0
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = x * x

    print('tape.gradient(y,x):', tape.gradient(y, x), '\n')
 

args.step = auto_increment(args.step, args.all)
### Step #13 - Getting a gradient of None: Took gradients through a stateful object
if args.step == 13:
    print("\n### Step #13 - Getting a gradient of None: Took gradients through a stateful object")

    x0 = tf.Variable(3.0)
    x1 = tf.Variable(0.0)

    with tf.GradientTape() as tape:
        # Update x1 = x1 + x0.
        x1.assign_add(x0)
        # The tape starts recording from x1.
        y = x1**2   # y = (x1 + x0)**2

    # This doesn't work.
    print(tape.gradient(y, x0), '\n')   #dy/dx0 = 2*(x1 + x0)    


args.step = auto_increment(args.step, args.all)
### Step #14 - No gradient registered
if args.step == 14:
    print("\n### Step #14 - No gradient registered")

    image = tf.Variable([[[0.5, 0.0, 0.0]]])
    delta = tf.Variable(0.1)

    with tf.GradientTape() as tape:
        new_image = tf.image.adjust_contrast(image, delta)

    try:
        print(tape.gradient(new_image, [image, delta]))
        assert False   # This should not happen.
    except LookupError as e:
        print(f'{type(e).__name__}: {e}\n')


args.step = auto_increment(args.step, args.all)
### Step #15 - Zeros instead of None
if args.step == 15:
    print("\n### Step #15 - Zeros instead of None")

    x = tf.Variable([2., 2.])
    y = tf.Variable(3.)

    with tf.GradientTape() as tape:
        z = y**2
    print(tape.gradient(z, x, unconnected_gradients=tf.UnconnectedGradients.ZERO), '\n')


### End of File
if args.plot:
    plt.show()
debug()
