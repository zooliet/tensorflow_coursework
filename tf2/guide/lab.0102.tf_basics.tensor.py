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
### Step #1 - Basics
if args.step == 1:
    print("\n### Step #1 - Basics")

    __doc__='''
    Tensors are multi-dimensional arrays with a uniform type (called a dtype).
    You can see all supported dtypes at tf.dtypes.DType. If you're familiar
    with NumPy, tensors are (kind of) like np.arrays. All tensors are
    immutable like Python numbers and strings: you can never update the
    contents of a tensor, only create a new one.
    '''
    print(__doc__)

    logger.info('Here is a "scalar" or "rank-0" tensor. A scalar contains a single value, and no "axes":')
    # This will be an int32 tensor by default
    rank_0_tensor = tf.constant(4)
    print(rank_0_tensor, '\n')

    logger.info('A "vector" or "rank-1" tensor is like a list of values. A vector has one axis:')
    rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
    print(rank_1_tensor, '\n')

    logger.info('A "matrix" or "rank-2" tensor has two axes:')
    rank_2_tensor = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float16)
    print(rank_2_tensor, '\n')

    logger.info('Tensors may have more axes; here is a tensor with three axes:')
    # There can be an arbitrary number of axes (sometimes called "dimensions")
    rank_3_tensor = tf.constant([
        [
            [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]
        ],
        [
            [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]
        ],
        [   
            [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]
        ],
    ])
    print(rank_3_tensor, '\n')

    if args.plot:
        plt.figure(figsize=(8,4))
        img = tf.io.read_file('supplement/tf2_g0102_01.png')
        img = tf.image.decode_png(img)
        plt.imshow(img)
        plt.show(block=False)

    logger.info('You can convert a tensor to a NumPy array either using np.array or the tensor.numpy method:')
    print(rank_2_tensor.numpy(), '\n')

    logger.info('You can do basic math on tensors, including addition, element-wise multiplication, and matrix multiplication:')
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[1, 1], [1, 1]]) # Could have also said `tf.ones([2,2])`
    print(tf.add(a, b), "\n")
    print(tf.multiply(a, b), "\n")
    print(tf.matmul(a, b), "\n")

    logger.info('Tensors are used in all kinds of operations (ops):')
    c = tf.constant([[4.0, 5.0], [10.0, 1.0]])
    # Find the largest value
    print('largest value:', tf.reduce_max(c), '\n')
    # Find the index of the largest value
    print('index of largest value:', tf.argmax(c), '\n')
    # Compute the softmax
    print('softmax:\n', tf.nn.softmax(c))


args.step = auto_increment(args.step, args.all)
### Step #2 - About shapes
if args.step == 2:
    print("\n### Step #2 - About shapes")

    __doc__='''
    Tensors have shapes. Some vocabulary:
    - Shape: The length (number of elements) of each of the axes of a tensor.
    - Rank: Number of tensor axes. A scalar has rank 0, a vector has rank 1, a
      matrix is rank 2.
    - Axis or Dimension: A particular dimension of a tensor.
    - Size: The total number of items in the tensor, the product shape vector.
    '''
    print(__doc__)

    rank_4_tensor = tf.zeros([3, 2, 4, 5])
    logger.info('rank_4_tensor = tf.zeros([3, 2, 4, 5])')
    print("Type of every element:", rank_4_tensor.dtype)
    print("Number of axes:", rank_4_tensor.ndim)
    print("Shape of tensor:", rank_4_tensor.shape)
    print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
    print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
    print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())

    if args.plot:
        plt.figure()
        plt.subplot(2,1,1)
        img = tf.io.read_file('supplement/tf2_g0102_02.png')
        img = tf.image.decode_png(img)
        plt.imshow(img)

        plt.subplot(2,1,2)
        img = tf.io.read_file('supplement/tf2_g0102_03.png')
        img = tf.image.decode_png(img)
        plt.imshow(img)

        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #3 - Indexing: Single-axis indexing
if args.step == 3:
    print("\n### Step #3 - Indexing: Single-axis indexing")

    logger.info('Indexing with a scalar removes the axis:')
    rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
    print(rank_1_tensor.numpy())
    print("First:", rank_1_tensor[0].numpy())
    print("Second:", rank_1_tensor[1].numpy())
    print("Last:", rank_1_tensor[-1].numpy(), '\n')

    logger.info('Indexing with a : slice keeps the axis:')
    print("Everything:", rank_1_tensor[:].numpy())
    print("Before 4:", rank_1_tensor[:4].numpy())
    print("From 4 to the end:", rank_1_tensor[4:].numpy())
    print("From 2, before 7:", rank_1_tensor[2:7].numpy())
    print("Every other item:", rank_1_tensor[::2].numpy())
    print("Reversed:", rank_1_tensor[::-1].numpy())


args.step = auto_increment(args.step, args.all)
### Step #4 - Indexing: Multi-axis indexing
if args.step == 4:
    print("\n### Step #4 - Indexing: Multi-axis indexing")

    rank_2_tensor = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float16)
    print(rank_2_tensor.numpy(), '\n')

    logger.info('Passing an integer for each index, the result is a scalar:')
    # Pull out a single value from a 2-rank tensor
    print(rank_2_tensor[1, 1].numpy(), '\n')

    logger.info('You can index using any combination of integers and slices:')
    # Get row and column tensors
    print("Second row:", rank_2_tensor[1, :].numpy())
    print("Second column:", rank_2_tensor[:, 1].numpy())
    print("Last row:", rank_2_tensor[-1, :].numpy())
    print("First item in last column:", rank_2_tensor[0, -1].numpy())
    print("Skip the first row:\n", rank_2_tensor[1:, :].numpy(), "\n")

    rank_3_tensor = tf.constant([
        [
            [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]
        ],
        [   
            [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]
        ],
        [
            [20, 21, 22, 23, 24],[25, 26, 27, 28, 29]
        ],
    ])
    logger.info('Here is an example with a 3-axis tensor:')
    print(rank_3_tensor[:, :, 4])

    if args.plot:
        plt.figure(figsize=(8,4))
        img = tf.io.read_file('supplement/tf2_g0102_04.png')
        img = tf.image.decode_png(img)
        plt.imshow(img)
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #5 - Indexing: Manipulating Shapes
if args.step == 5:
    print("\n### Step #5 - Indexing: Manipulating Shapes")

    logger.info('Shape returns a `TensorShape` object that shows the size along each axis:')
    x = tf.constant([[1], [2], [3]])
    print(x.shape)

    logger.info('You can convert this object into a Python list, too:')
    print(x.shape.as_list())

    logger.info('You can reshape a tensor into a new shape:')
    # Note that you're passing in a list
    reshaped = tf.reshape(x, [1, 3])
    print(x.shape, '=>', reshaped.shape, '\n')

    rank_3_tensor = tf.constant([
        [[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9]],
        [[10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19]],
        [[20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29]],])
    print(rank_3_tensor, '\n')

    logger.info('A `-1` passed in the `shape` argument says "Whatever fits":')
    print(tf.reshape(rank_3_tensor, [-1]), '\n')

    logger.info('For this 3x2x5 tensor, reshaping to (3x2)x5 or 3x(2x5) are both reasonable things to do:')
    print(tf.reshape(rank_3_tensor, [3*2, 5]), "\n")
    print(tf.reshape(rank_3_tensor, [3, -1]), '\n')

    logger.info("Bad examples: don't do this:")
    # You can't reorder axes with reshape.
    print(tf.reshape(rank_3_tensor, [2, 3, 5]), "\n")

    # This is a mess
    print(tf.reshape(rank_3_tensor, [5, 6]), "\n")

    # This doesn't work at all
    try:
      tf.reshape(rank_3_tensor, [7, -1])
    except Exception as e:
      print(f"{type(e).__name__}: {e}")

    if args.plot:
        plt.figure()
        plt.subplot(2, 1, 1)
        img = tf.io.read_file('supplement/tf2_g0102_05.png')
        img = tf.image.decode_png(img)
        plt.imshow(img)

        plt.subplot(2, 1, 2)
        img = tf.io.read_file('supplement/tf2_g0102_06.png')
        img = tf.image.decode_png(img)
        plt.imshow(img)
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #6 - More on DTypes
if args.step == 6:
    print("\n### Step #6 - More on DTypes")

    logger.info('You can cast from type to type:')
    the_f64_tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float64)
    the_f16_tensor = tf.cast(the_f64_tensor, dtype=tf.float16)
    # Now, cast to an uint8 and lose the decimal precision
    the_u8_tensor = tf.cast(the_f16_tensor, dtype=tf.uint8)
    print(the_u8_tensor)


args.step = auto_increment(args.step, args.all)
### Step #7 - Broadcasting
if args.step == 7:
    print("\n### Step #7 - Broadcasting")

    __doc__='''
    Broadcasting is a concept borrowed from the equivalent feature in NumPy. In
    short, under certain conditions, smaller tensors are "stretched"
    automatically to fit larger tensors when running combined operations on
    them.
    '''
    print(__doc__)

    logger.info('the scalar is broadcast to be the same shape as the other argument:')
    x = tf.constant([1, 2, 3])
    y = tf.constant(2)
    z = tf.constant([2, 2, 2])
    # All of these are the same computation
    print(tf.multiply(x, 2))
    print(x * y)
    print(x * z)
    print()

    logger.info('A broadcasted add: a [2,1] times a [1,3] gives a [2,3]:')
    x = tf.constant([1, 2], name='x')
    x = tf.reshape(x, [2,1])
    y = tf.constant([3,4,5])
    logger.info(f'x:\n{x}')
    logger.info(f'y:\n{y}')
    logger.info(f'x*y:\n{tf.multiply(x,y)}\n')

    logger.info('Here is the same operation without broadcasting:')
    x_stretch = tf.constant([[1,1,1],[2,2,2]])
    y_stretch = tf.constant([[3,4,5],[3,4,5]])
    logger.info(f'x_stretch:\n{x_stretch}')
    logger.info(f'y_stretch:\n{y_stretch}')
    logger.info(f'x_stretch*y_stretch:\n{tf.multiply(x_stretch,y_stretch)}\n')

    # logger.info('A broadcasted add: a [3, 1] times a [1, 4] gives a [3,4]:')
    # # These are the same computations
    # x = tf.reshape(x,[3,1])
    # y = tf.range(1, 5)
    # print(x, "\n")
    # print(y, "\n")
    # print(tf.multiply(x, y), '\n')
    #
    # x_stretch = tf.constant([[1, 1, 1, 1],
    #                         [2, 2, 2, 2],
    #                         [3, 3, 3, 3]])
    #
    # y_stretch = tf.constant([[1, 2, 3, 4],
    #                          [1, 2, 3, 4],
    #                          [1, 2, 3, 4]])
    #
    # print(x_stretch * y_stretch)  # Again, operator overloading

    logger.info('You see what broadcasting looks like using tf.broadcast_to:')
    print(tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3]))


args.step = auto_increment(args.step, args.all)
### Step #8 - tf.convert_to_tensor
if args.step == 8:
    print("\n### Step #8 - tf.convert_to_tensor")

    __doc__='''
    Most ops, like tf.matmul and tf.reshape take arguments of class tf.Tensor.
    However, you'll notice in the above case, Python objects shaped like
    tensors are accepted.  Most, but not all, ops call convert_to_tensor on
    non-tensor arguments. There is a registry of conversions, and most object
    classes like NumPy's ndarray, TensorShape, Python lists, and tf.Variable
    will all convert automatically.
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #9 - Ragged Tensors
if args.step == 9:
    print("\n### Step #9 - Ragged Tensors")

    __doc__='''
    A tensor with variable numbers of elements along some axis is called
    "ragged". Use tf.ragged.RaggedTensor for ragged data.
    '''
    print(__doc__)

    ragged_list = [
        [0, 1, 2, 3],
        [4, 5],
        [6, 7, 8],
        [9]
    ]

    logger.info('This cannot be represented as a regular tensor:')
    try: 
        tensor = tf.constant(ragged_list)
    except Exception as e:
        print(f"{type(e).__name__}: {e}\n")

    logger.info('create a tf.RaggedTensor using tf.ragged.constant:')
    ragged_tensor = tf.ragged.constant(ragged_list)
    print(ragged_tensor, '\n')

    logger.info('The shape of a tf.RaggedTensor contains some axes with unknown lengths:')
    print(ragged_tensor.shape)

    if args.plot:
        plt.figure(figsize=(8,4))
        img = tf.io.read_file('supplement/tf2_g0102_07.png')
        img = tf.image.decode_png(img)
        plt.imshow(img)
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #10 - String tensors
if args.step == 10:
    print("\n### Step #10 - String tensors")

    __doc__='''
    tf.string is a dtype, which is to say you can represent data as strings
    (variable-length byte arrays) in tensors.
    '''
    print(__doc__)

    logger.info('Tensors can be strings, too. here is a scalar string:')
    scalar_string_tensor = tf.constant("Gray wolf", dtype=tf.string)
    print(scalar_string_tensor, '\n')

    logger.info('vector of strings:')
    # If you have three string tensors of different lengths, this is OK.
    tensor_of_strings = tf.constant(["Gray wolf", "Quick brown fox", "Lazy dog"])
    # Note that the shape is (3,). The string length is not included.
    print(tensor_of_strings, '\n')

    __doc__='''
    In the above printout the b prefix indicates that tf.string dtype is not a
    unicode string, but a byte-string.
    '''
    print(__doc__)

    logger.info('If you pass unicode characters they are utf-8 encoded:')
    print(tf.constant("🥳👍"), '\n')

    logger.info('Some basic functions with strings can be found in tf.strings:')
    # You can use split to split a string into a set of tensors
    print(tf.strings.split(scalar_string_tensor, sep=" "), '\n')

    # ...but it turns into a `RaggedTensor` if you split up a tensor of strings,
    # as each string might be split into a different number of parts.
    print(tf.strings.split(tensor_of_strings), '\n')

    text = tf.constant("1 10 100")
    print(tf.strings.to_number(tf.strings.split(text, " ")), '\n')

    # Although you can't use tf.cast to turn a string tensor into numbers, 
    # you can convert it into bytes, and then into numbers.
    logger.info('String to number:')
    byte_strings = tf.strings.bytes_split(tf.constant("Duck"))
    byte_ints = tf.io.decode_raw(tf.constant("Duck"), tf.uint8)
    print("Byte strings:", byte_strings)
    print("Bytes:", byte_ints, '\n')

    logger.info('split it up as unicode and then decode it:')
    unicode_bytes = tf.constant("アヒル 🦆")
    unicode_char_bytes = tf.strings.unicode_split(unicode_bytes, "UTF-8")
    unicode_values = tf.strings.unicode_decode(unicode_bytes, "UTF-8")

    print("\nUnicode bytes:", unicode_bytes)
    print("\nUnicode chars:", unicode_char_bytes)
    print("\nUnicode values:", unicode_values)


args.step = auto_increment(args.step, args.all)
### Step #11 - Sparse tensors
if args.step == 11:
    print("\n### Step #11 - Sparse tensors")

    __doc__='''
    Sometimes, your data is sparse, like a very wide embedding space.
    TensorFlow supports tf.sparse.SparseTensor and related operations to store
    sparse data efficiently.
    '''
    print(__doc__)

    logger.info('Sparse tensors store values by index in a memory-efficient manner:')
    sparse_tensor = tf.sparse.SparseTensor(
        indices=[[0, 0], [1, 2]],
        values=[1, 2],
        dense_shape=[3, 4]
    )
    print(sparse_tensor, "\n")

    logger.info('You can convert sparse tensors to dense:')
    print(tf.sparse.to_dense(sparse_tensor))

    if args.plot:
        plt.figure()
        img = tf.io.read_file('supplement/tf2_g0102_08.png')
        img = tf.image.decode_png(img)
        plt.imshow(img)
        plt.show(block=False)


### End of File
print()
if args.plot:
    plt.show()
debug()
