#!/usr/bin/env python

import sys
sys.path.append('./')
sys.path.append('../../')

from lab_utils import (
    tf, os, np, plt, logger, ap, BooleanAction,
    debug, toc, auto_increment
)

# ap.add_argument('--epochs', type=int, default=2, help='number of epochs: 2*')
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
import pandas as pd
import pathlib

from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import Layer, Dense, Flatten


### TOC
if args.step == 0:
    toc(__file__)


if args.step or args.all:
    flowers_root = tf.keras.utils.get_file(
        'flower_photos',
        'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
        untar=True
    )


args.step = auto_increment(args.step, args.all)
### Step #1 - Introduction
if args.step == 1:
    print("\n### Step #1 - Introduction")
    
    __doc__= '''
    The tf.data API enables you to build complex input pipelines from simple,
    reusable pieces. For example, the pipeline for an image model might aggregate
    data from files in a distributed file system, apply random perturbations to
    each image, and merge randomly selected images into a batch for training. The
    pipeline for a text model might involve extracting symbols from raw text data,
    converting them to embedding identifiers with a lookup table, and batching
    together sequences of different lengths. The tf.data API makes it possible to
    handle large amounts of data, read from different data formats, and perform
    complex transformations.

    The tf.data API introduces a tf.data.Dataset abstraction that represents a
    sequence of elements, in which each element consists of one or more components.
    For example, in an image pipeline, an element might be a single training
    example, with a pair of tensor components representing the image and its label.

    There are two distinct ways to create a dataset:
    - A data source constructs a Dataset from data stored in memory or in one or more
      files.
    - A data transformation constructs a dataset from one or more tf.data.Dataset
      objects.
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #2 - Basic mechanics
if args.step == 2:
    print("\n### Step #2 - Basic mechanics")

    __doc__='''
    To create an input pipeline, you must start with a data source. For
    example, to construct a Dataset from data in memory, you can use
    tf.data.Dataset.from_tensors() or tf.data.Dataset.from_tensor_slices().

    Alternatively, if your input data is stored in a file in the recommended
    TFRecord format, you can use tf.data.TFRecordDataset().

    Once you have a Dataset object, you can transform it into a new Dataset by
    chaining method calls on the tf.data.Dataset object. For example, you can
    apply per-element transformations such as Dataset.map(), and multi-element
    transformations such as Dataset.batch().

    The Dataset object is a Python iterable. This makes it possible to consume
    its elements using a for loop.
    '''
    print(__doc__)

    dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])
    logger.info(f'dataset.element_spec:\n{dataset.element_spec}')
    print(*list(iter(dataset)), sep='\n')
    logger.info(f'dataset.reduce(): {dataset.reduce(0, lambda state, value: state + value).numpy()}')


args.step = auto_increment(args.step, args.all)
### Step #3 - Basic mechanics: Dataset structure
if args.step == 3:
    print("\n### Step #3 - Basic mechanics: Dataset structure")

    __doc__='''
    A dataset produces a sequence of elements, where each element is the same
    (nested) structure of components. Individual components of the structure
    can be of any type representable by tf.TypeSpec, including tf.Tensor,
    tf.sparse.SparseTensor, tf.RaggedTensor, tf.TensorArray, or
    tf.data.Dataset.

    The Python constructs that can be used to express the (nested) structure of
    elements include tuple, dict, NamedTuple, and OrderedDict. In particular,
    list is not a valid construct for expressing the structure of dataset
    elements. This is because early tf.data users felt strongly about list
    inputs (e.g. passed to tf.data.Dataset.from_tensors) being automatically
    packed as tensors and list outputs (e.g. return values of user-defined
    functions) being coerced into a tuple. As a consequence, if you would like
    a list input to be treated as a structure, you need to convert it into
    tuple and if you would like a list output to be a single component, then
    you need to explicitly pack it using tf.stack.

    The Dataset.element_spec property allows you to inspect the type of each
    element component. The property returns a nested structure of tf.TypeSpec
    objects, matching the structure of the element, which may be a single
    component, a tuple of components, or a nested tuple of components. 
    '''
    print(__doc__)

    dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 6]))
    logger.info(f'dataset1.element_spec:\n{dataset1.element_spec}\n')
    print(*list(iter(dataset1)), sep='\n')
    print()

    dataset2 = tf.data.Dataset.from_tensor_slices( 
        (tf.random.uniform([4]), 
         tf.random.uniform([4, 100], maxval=100, dtype=tf.int32))
    )
    logger.info('dataset2.element_spec:')
    print(*dataset2.element_spec, sep='\n')
    print()

    dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
    logger.info('dataset3.element_spec:')
    print(*dataset3.element_spec, sep='\n')
    print()
    for a, (b,c) in dataset3:
        logger.info('shapes: {a.shape}, {b.shape}, {c.shape}'.format(a=a, b=b, c=c))
        break
    print()

    # Dataset containing a sparse tensor.
    sparse_tensor = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
    logger.info(f'sparse_tensor\n{tf.sparse.to_dense(sparse_tensor)}\n')
    dataset4 = tf.data.Dataset.from_tensors(sparse_tensor)
    logger.info(f'dataset4.element_spec:\n{dataset4.element_spec}')
    logger.info(f'dataset4.element_spec.value_type:\n{dataset4.element_spec.value_type}')
    # Use value_type to see the type of value represented by the element spec


args.step = auto_increment(args.step, args.all)
### Step #4 - Reading input data: Consuming NumPy arrays
if args.step == 4:
    print("\n### Step #4 - Reading input data: Consuming NumPy arrays")

    __doc__='''
    If all of your input data fits in memory, the simplest way to create a
    Dataset from them is to convert them to tf.Tensor objects and use
    Dataset.from_tensor_slices()
    '''
    print(__doc__)

    train, test = tf.keras.datasets.fashion_mnist.load_data()
    images, labels = train
    images = images/255

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    logger.info('dataset.element_spec:')
    print(*dataset.element_spec, sep='\n')
    print()
    for image, label in dataset.take(1):
        logger.info(f"image's shape: {image.shape}, label's shape: {label.shape}")
    for image_batch, label_batch in dataset.batch(32).take(1):
        logger.info(f"image_batch's shape: {image_batch.shape}, label_batch's shape: {label_batch.shape}")


args.step = auto_increment(args.step, args.all)
### Step #5 - Reading input data: Consuming Python generators
if args.step == 5:
    print("\n### Step #5 - Reading input data: Consuming Python generators")

    def count(stop):
        i = 0
        while i<stop:
            yield i
            i += 1
    for n in count(5):
        logger.debug(n)

    ds_counter = tf.data.Dataset.from_generator(
        count, args=[25], output_types=tf.int32, output_shapes = (), 
    )

    logger.info('ds_counter.batch(10).take(3):')
    for count_batch in ds_counter.batch(10).take(3):
        print(count_batch.numpy())
    print()

    logger.info('ds_counter.repeat().batch(10).take(3):')
    for count_batch in ds_counter.repeat().batch(10).take(3):
        print(count_batch.numpy())
    print()

    def gen_series():
        i = 0
        while True:
            size = np.random.randint(0, 10)
            yield i, np.random.normal(size=(size,))
            i += 1

    for i, series in gen_series():
        logger.debug(f'{i}:{str(series)}')
        if i > 5:
            break

    ds_series = tf.data.Dataset.from_generator(
        gen_series,
        output_types=(tf.int32, tf.float32),
        output_shapes=((), (None,))
    )

    logger.info('ds_series.padded_batch(3).take(2):')
    for ids, sequence_batch in ds_series.padded_batch(3).take(2):
        for id, sequence in zip(ids, sequence_batch):
            print(f'{id.numpy()}: {sequence.numpy()}')
        print()

    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, rotation_range=20
    )

    image_iters = img_gen.flow_from_directory(flowers_root, batch_size=32)
    images_batch, labels_batch = next(image_iters)
    logger.debug(f'{images_batch.dtype}, {images_batch.shape}')
    logger.debug(f'{labels_batch.dtype}, {labels_batch.shape}')

    ds = tf.data.Dataset.from_generator(
        lambda: img_gen.flow_from_directory(flowers_root),
        output_types=(tf.float32, tf.float32),
        output_shapes=([32,256,256,3], [32,5])
    )
    logger.info('ds.element_spec:')
    print(*ds.element_spec, sep='\n')
    print()
    
    for images_batch, labels_batch in ds.take(1):
        logger.info(f'images_batch.shape: {images_batch.shape}')
        logger.info(f'labels_batch.shape: {labels_batch.shape}')


args.step = auto_increment(args.step, args.all)
### Step #6 - Reading input data: Consuming TFRecord data
if args.step == 6:
    print("\n### Step #6 - Reading input data: Consuming TFRecord data")

    __doc__='''
    The tf.data API supports a variety of file formats so that you can process
    large datasets that do not fit in memory. For example, the TFRecord file
    format is a simple record-oriented binary format that many TensorFlow
    applications use for training data. The tf.data.TFRecordDataset class
    enables you to stream over the contents of one or more TFRecord files as
    part of an input pipeline.  
    '''
    print(__doc__)

    # Creates a dataset that reads all of the examples from two files.
    # French Street Name Signs (FSNS).
    fsns_test_file = tf.keras.utils.get_file(
        "fsns.tfrec", 
        "https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001"
    )
    dataset = tf.data.TFRecordDataset(filenames = [fsns_test_file])
    logger.info('dataset.element_spec:')
    print(dataset.element_spec, '\n')
    
    # take one serialized tf.train.Example record
    raw_example = next(iter(dataset))
    # decode
    parsed = tf.train.Example.FromString(raw_example.numpy())
    logger.info('decoded by tf.train.Example.FromString():')
    print(parsed.features.feature['image/text'])


args.step = auto_increment(args.step, args.all)
### Step #7 - Reading input data: Consuming text data
if args.step == 7:
    print("\n### Step #7 - Reading input data: Consuming text data")

    __doc__='''
    Many datasets are distributed as one or more text files. The
    tf.data.TextLineDataset provides an easy way to extract lines from one or
    more text files. Given one or more filenames, a TextLineDataset will
    produce one string-valued element per line of those files.
    '''
    print(__doc__)

    directory_url = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
    file_names = ['cowper.txt', 'derby.txt', 'butler.txt']

    file_paths = [
        tf.keras.utils.get_file(file_name, directory_url + file_name)
        for file_name in file_names
    ]

    dataset = tf.data.TextLineDataset(file_paths)
    logger.info('dataset.take(5):')
    for line in dataset.take(5):
        print(line.numpy())
    print()

    #
    files_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    lines_ds = files_ds.interleave(tf.data.TextLineDataset, cycle_length=3)

    logger.info('line_ds.take(9):')
    for i, line in enumerate(lines_ds.take(9)):
      if i % 3 == 0:
        tf.print()
      print(line.numpy())
    print()

    #
    titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
    titanic_lines = tf.data.TextLineDataset(titanic_file)
    logger.info('titanic_lines.take(5):')
    for line in titanic_lines.take(5):
        print(line.numpy())
    print()

    # 0th 위치, 1길이 값이 "0"이 아닌 라인
    def survived(line):
        return tf.not_equal(tf.strings.substr(line, 0, 1), "0") 

    survivors = titanic_lines.skip(1).filter(survived) # remove the header line
    logger.info('titanic_lines.skip(1).filter(survived).take(5):')
    for line in survivors.take(5):
        print(line.numpy())
    

args.step = auto_increment(args.step, args.all)
### Step #8 - Reading input data: Consuming CSV data
if args.step == 8:
    print("\n### Step #8 - Reading input data: Consuming CSV data")

    titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
    df = pd.read_csv(titanic_file)
    logger.info(f'df.head():\n{df.head()}\n')

    # in-memory
    titanic_slices = tf.data.Dataset.from_tensor_slices(dict(df))
    logger.info('titanic_slices.element_spec:')
    print(*titanic_slices.element_spec.items(), sep='\n')
    print()
    logger.info('titanic_slices.batch(2).take(2):')
    for feature_batch in titanic_slices.batch(2).take(2):
        for key, value in feature_batch.items():
            print("{!r:20s}: {}".format(key, value))
        print()

    # from disk
    titanic_batches = tf.data.experimental.make_csv_dataset(
        titanic_file, 
        batch_size=4,
        label_name="survived"
    )
    logger.info(f'len(titanic_batches.element_spec): {len(titanic_batches.element_spec)}\n') # features, label
    logger.info('titanic_batches.element_spec[0] - features:')
    print(*(dict(titanic_batches.element_spec[0]).items()), sep='\n')
    print()
    logger.info('titanic_batches.element_spec[1] - label:')
    print(titanic_batches.element_spec[1])
    print()

    logger.info('titanic_batches.take(2): batch=4') # batch = 4
    for feature_batch, label_batch in titanic_batches.take(2):
        print("{!r:20s}: {}".format('label(survived)', label_batch))
        # logger.info("features:")
        for key, value in feature_batch.items():
            print("{!r:20s}: {}".format(key, value))
        print()

    # from disk: select columns
    titanic_batches2 = tf.data.experimental.make_csv_dataset(
        titanic_file, 
        batch_size=4,
        label_name="survived", 
        select_columns=['class', 'fare', 'survived']
    )

    logger.info('titanic_batches2.take(2): batch=4') # batch = 4
    for feature_batch, label_batch in titanic_batches2.take(2):
        print("{!r:20s}: {}".format('label(survived)', label_batch))
        # logger.info("features:")
        for key, value in feature_batch.items():
            print("{!r:20s}: {}".format(key, value))
        print()

    # lower-level
    titanic_types  = [
        tf.int32, tf.string, tf.float32, tf.int32, tf.int32, 
        tf.float32, tf.string, tf.string, tf.string, tf.string
    ]
    dataset = tf.data.experimental.CsvDataset(
        titanic_file, 
        titanic_types , 
        header=True
    )

    logger.info('tf.data.experimental.CsvDataset().take(4):')
    for line in dataset.take(4): 
        print([item.numpy() for item in line])


args.step = auto_increment(args.step, args.all)
### Step #9 - Reading input data: Consuming sets of files
if args.step == 9:
    print("\n### Step #9 - Reading input data: Consuming sets of files")

    flowers_root = pathlib.Path(flowers_root)
    logger.info(f'tf.data.Dataset.list_files({str(flowers_root)}):')
    list_ds = tf.data.Dataset.list_files(str(flowers_root/'*/*'))
    for f in list_ds.take(5):
        print(f.numpy())
    print()

    def process_path(file_path):
        label = tf.strings.split(file_path, os.sep)[-2]
        return tf.io.read_file(file_path), label

    labeled_ds = list_ds.map(process_path)
    logger.info('labeled_ds.element_spec:')
    print(*labeled_ds.element_spec, sep='\n')
    print()

    logger.info('labeled_ds.take(2):')
    for image_raw, label_text in labeled_ds.take(2):
        print(label_text.numpy())
        print(repr(image_raw.numpy()[:100]), '\n')


args.step = auto_increment(args.step, args.all)
### Step #10 - Batching dataset elements: Simple batching
if args.step == 10:
    print("\n### Step #10 - Batching dataset elements: Simple batching")

    inc_dataset = tf.data.Dataset.range(20)
    dec_dataset = tf.data.Dataset.range(0, -20, -1)
    dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
    logger.info('dataset.element_spec:')
    print(*dataset.element_spec, '\n')

    batched_dataset = dataset.batch(4)
    logger.info('dataset.batch(4).take(3):')
    for batch in batched_dataset.take(3):
        # print(*[arr for arr in batch], sep='\n')
        print(*batch, sep='\n\n')
    print()

    # using drop_remainder
    logger.info('dataset.batch(7, drop_remainder=True):')
    batched_dataset = dataset.batch(7, drop_remainder=True)
    for batch in batched_dataset:
        print(*batch, sep='\n\n')


args.step = auto_increment(args.step, args.all)
### Step #11 - Batching dataset elements: Batching tensors with padding
if args.step == 11:
    print("\n### Step #11 - Batching dataset elements: Batching tensors with padding")

    dataset = tf.data.Dataset.range(100)
    dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
    # dataset = dataset.padded_batch(4)
    dataset = dataset.padded_batch(4, padded_shapes=(None,))

    logger.info('dataset.batch(4).take(3):')
    for i, batch in enumerate(dataset.take(3)):
        print(f'{i+1}th batch:\n{batch.numpy()}')


args.step = auto_increment(args.step, args.all)
### Step #12 - Training workflows: Processing multiple epochs
if args.step == 12:
    print("\n### Step #12 - Training workflows: Processing multiple epochs")

    titanic_file = tf.keras.utils.get_file(
        "train.csv", 
        "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
    )
    titanic_lines = tf.data.TextLineDataset(titanic_file)

    def plot_batch_sizes(ds, title=None):
        batch_sizes = [batch.shape[0] for batch in ds]
        plt.figure()
        plt.bar(range(len(batch_sizes)), batch_sizes)
        plt.title(title)
        plt.xlabel('Batch number')
        plt.ylabel('Batch size')
        plt.show(block=False)

    if args.plot:
        titanic_batches = titanic_lines.batch(128)
        plot_batch_sizes(titanic_batches, 'ds.batch(128)')

        titanic_batches = titanic_lines.repeat(3).batch(128)
        plot_batch_sizes(titanic_batches, 'ds.repeat(3).batch(128)')

        titanic_batches = titanic_lines.batch(128).repeat(3)
        plot_batch_sizes(titanic_batches, 'ds.batch(128).repeat(3)')

    epochs = 3
    dataset = titanic_lines.batch(128)
    
    logger.info('ds.batch(128) for 3 epochs:')
    for epoch in range(epochs):
        for batch in dataset:
            print(batch.shape)
        logger.info("End of epoch: %d" % epoch)


args.step = auto_increment(args.step, args.all)
### Step #13 - Training workflows: Randomly shuffling input data
if args.step == 13:
    print("\n### Step #13 - Training workflows: Randomly shuffling input data")

    titanic_file = tf.keras.utils.get_file(
        "train.csv", 
        "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
    )
    lines = tf.data.TextLineDataset(titanic_file)
    counter = tf.data.experimental.Counter()

    dataset = tf.data.Dataset.zip((counter, lines))
    dataset = dataset.shuffle(buffer_size=30)
    dataset = dataset.batch(10)
    logger.info('ds.shuffle(buffer_size=30).batch(10):')
    for n, line_batch in dataset.take(3):
        print(n.numpy()) 
    print()

    #
    dataset = tf.data.Dataset.zip((counter, lines))
    shuffled = dataset.shuffle(buffer_size=100).batch(10).repeat(2)
    shuffle_repeat = [n.numpy().mean() for n, line_batch in shuffled]

    # Here are the item ID's near the epoch boundary
    logger.info('ds.shuffle(buffer_size=100).batch(10).repeat(2):')
    for n, line_batch in shuffled.skip(60).take(5):
        print(n.numpy()) 
    print()

    dataset = tf.data.Dataset.zip((counter, lines))
    shuffled = dataset.repeat(2).shuffle(buffer_size=100).batch(10)
    repeat_shuffle = [n.numpy().mean() for n, line_batch in shuffled]

    logger.info('ds.repeat(2).shuffle(buffer_size=100).batch(10):')
    for n, line_batch in shuffled.skip(55).take(15):
        print(n.numpy()) 

    if args.plot:
        plt.figure()
        plt.plot(shuffle_repeat, label="shuffle().repeat()")
        plt.plot(repeat_shuffle, label="repeat().shuffle()")
        plt.ylabel("Mean item ID")
        plt.legend()
        plt.show(block=False)


args.step = auto_increment(args.step, args.all)
### Step #14 - Preprocessing data
if args.step == 14: 
    print("\n### Step #14 - Preprocessing data")

    __doc__='''
    The Dataset.map(f) transformation produces a new dataset by applying a
    given function f to each element of the input dataset. It is based on the
    map() function that is commonly applied to lists (and other structures) in
    functional programming languages. The function f takes the tf.Tensor
    objects that represent a single element in the input, and returns the
    tf.Tensor objects that will represent a single element in the new dataset.
    Its implementation uses standard TensorFlow operations to transform one
    element into another.
    '''
    print(__doc__)


args.step = auto_increment(args.step, args.all)
### Step #15 - Preprocessing data: Decoding image data and resizing it
if args.step in [15, 16]: 
    print("\n### Step #15 - Preprocessing data: Decoding image data and resizing it")
    
    # Reads an image from a file, decodes it into a dense tensor, and resizes it
    # to a fixed shape.
    def parse_image(filename):
        parts = tf.strings.split(filename, os.sep)
        label = parts[-2]

        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [128, 128])
        return image, label

    def show(image, label):
        plt.figure()
        plt.imshow(image)
        plt.title(label.numpy().decode('utf-8'))
        plt.axis('off')
        plt.show(block=False)

    flowers_root = pathlib.Path(flowers_root)
    list_ds = tf.data.Dataset.list_files(str(flowers_root/'*/*'))

    # file_path = next(iter(list_ds))
    # image, label = parse_image(file_path)
    # if args.plot:
    #     show(image, label)

    images_ds = list_ds.map(parse_image)

    if args.plot and args.step == 15:
        for image, label in images_ds.take(5):
            show(image, label)


args.step = auto_increment(args.step, args.all)
### Step #16 - Preprocessing data: Applying arbitrary Python logic 
if args.step == 16:
    print("\n### Step #16 - Preprocessing data: Applying arbitrary Python logic")

    import scipy.ndimage as ndimage
    def random_rotate_image(image):
        image = ndimage.rotate(image, np.random.uniform(-30, 30), reshape=False)
        return image

    flowers_root = pathlib.Path(flowers_root)
    list_ds = tf.data.Dataset.list_files(str(flowers_root/'*/*'))
    images_ds = list_ds.map(parse_image)

    image, label = next(iter(images_ds))
    image = random_rotate_image(image)
    if args.plot:
        show(image, label)

    def tf_random_rotate_image(image, label):
        im_shape = image.shape
        [image,] = tf.py_function(random_rotate_image, [image], [tf.float32])
        image.set_shape(im_shape)
        return image, label

    rot_ds = images_ds.map(tf_random_rotate_image)
    if args.plot:
        for image, label in rot_ds.take(5):
            show(image, label)


args.step = auto_increment(args.step, args.all)
### Step #17 - Preprocessing data: Parsing tf.Example protocol buffer messages
if args.step == 17:
    print("\n### Step #17 - Preprocessing data: Parsing tf.Example protocol buffer messages")

    __doc__='''
    Many input pipelines extract tf.train.Example protocol buffer messages from
    a TFRecord format. Each tf.train.Example record contains one or more
    "features", and the input pipeline typically converts these features into
    tensors.
    '''
    print(__doc__)

    fsns_test_file = tf.keras.utils.get_file(
        "fsns.tfrec", 
        "https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001"
    )

    dataset = tf.data.TFRecordDataset(filenames = [fsns_test_file])

    raw_example = next(iter(dataset))
    parsed = tf.train.Example.FromString(raw_example.numpy())

    feature = parsed.features.feature
    raw_img = feature['image/encoded'].bytes_list.value[0]

    if args.plot:
        plt.figure()
        img = tf.image.decode_png(raw_img)
        plt.imshow(img)
        plt.axis('off')
        _ = plt.title(feature["image/text"].bytes_list.value[0])
        plt.show(block=False) 
    
    def tf_parse(eg):
        example = tf.io.parse_example(
            eg[tf.newaxis], {
                'image/encoded': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
                'image/text': tf.io.FixedLenFeature(shape=(), dtype=tf.string)
            }
        )
        return example['image/encoded'][0], example['image/text'][0]

    img, txt = tf_parse(raw_example)
    logger.info('tf_parse(raw_example):')
    print(txt.numpy())
    print(repr(img.numpy()[:20]), '\n')

    decoded = dataset.map(tf_parse)
    image_batch, text_batch = next(iter(decoded.batch(10)))
    logger.info('dataset.map(tf_parse):')
    print(*text_batch, sep='\n')


args.step = auto_increment(args.step, args.all)
### Step #18 - Preprocessing data: Time series windowing
if args.step == 18:
    print("\n### Step #18 - Preprocessing data: Time series windowing")

    range_ds = tf.data.Dataset.range(100000)
    batches = range_ds.batch(10, drop_remainder=True)

    logger.info('batches.take(5): when batch_size is 10')
    for batch in batches.take(5):
        print(batch.numpy())
    print()

    #
    def dense_1_step(batch):
        # Shift features and labels one step relative to each other.
        return batch[:-1], batch[1:]

    predict_dense_1_step = batches.map(dense_1_step)

    logger.info('predict_dense_1_step.take(3):')
    for features, label in predict_dense_1_step.take(3):
        print(features.numpy(), " => ", label.numpy())
    print()

    #
    batches = range_ds.batch(15, drop_remainder=True)

    def label_next_5_steps(batch):
        return (
            batch[:-5],   # Inputs: All except the last 5 steps
            batch[-5:]
        )   # Labels: The last 5 steps

    predict_5_steps = batches.map(label_next_5_steps)

    logger.info('predict_5_steps.take(3):')
    for features, label in predict_5_steps.take(3):
        print(features.numpy(), " => ", label.numpy())
    print()

    #
    feature_length = 10
    label_length = 3

    features = range_ds.batch(feature_length, drop_remainder=True)
    labels = range_ds.batch(feature_length).skip(1).map(lambda labels: labels[:label_length])

    predicted_steps = tf.data.Dataset.zip((features, labels))

    logger.info('predicted_steps.take(5):')
    for features, label in predicted_steps.take(5):
        print(features.numpy(), " => ", label.numpy())
    print()

    __doc__='''
    While using Dataset.batch works, there are situations where you may need
    finer control. The Dataset.window method gives you complete control, but
    requires some care: it returns a Dataset of Datasets.
    '''
    print(__doc__)

    window_size = 5

    windows = range_ds.window(window_size, shift=1)
    logger.info('window.take(5):')
    print(*windows.take(5), sep='\n')
    # for sub_ds in windows.take(5):
    #     print(sub_ds)

    __doc__='''
    The Dataset.flat_map method can take a dataset of datasets and flatten it
    into a single dataset.
    ''' 
    print(__doc__)

    logger.info('windows.flat_map(lambda x: x).take(30):')
    for x in windows.flat_map(lambda x: x).take(30):
        print(x.numpy(), end=" ")
    print('\n')

    # 
    def sub_to_batch(sub):
        return sub.batch(window_size, drop_remainder=True)

    logger.info('windows.flat_map(sub_to_batch).take(30):')
    for example in windows.flat_map(sub_to_batch).take(5):
        print(example.numpy())
    print()

    #
    def make_window_dataset(ds, window_size=5, shift=1, stride=1):
        windows = ds.window(window_size, shift=shift, stride=stride)

        def sub_to_batch(sub):
            return sub.batch(window_size, drop_remainder=True)

        windows = windows.flat_map(sub_to_batch)
        return windows

    ds = make_window_dataset(range_ds, window_size=10, shift = 5, stride=3)

    for example in ds.take(10):
        print(example.numpy())
    print()

    dense_labels_ds = ds.map(dense_1_step)

    for inputs,labels in dense_labels_ds.take(3):
        print(inputs.numpy(), "=>", labels.numpy())


args.step = auto_increment(args.step, args.all)
### Step #19 - Preprocessing data: Resampling
if args.step == 19:
    print("\n### Step #19 - Preprocessing data: Resampling")

    __doc__='''
    When working with a dataset that is very class-imbalanced, you may want to
    resample the dataset. tf.data provides two methods to do this.
    '''
    print(__doc__)

    zip_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/download.tensorflow.org/data/creditcard.zip',
        fname='creditcard.zip',
        extract=True
    )
    csv_path = zip_path.replace('.zip', '.csv')

    creditcard_ds = tf.data.experimental.make_csv_dataset(
        csv_path,
        batch_size=1024, 
        label_name="Class",
        # Set the column types: 30 floats and an int.
        column_defaults=[float()]*30+[int()]
    )

    # Now, check the distribution of classes, it is highly skewed:
    def count(counts, batch):
        features, labels = batch
        class_1 = labels == 1
        class_1 = tf.cast(class_1, tf.int32)

        class_0 = labels == 0
        class_0 = tf.cast(class_0, tf.int32)

        counts['class_0'] += tf.reduce_sum(class_0)
        counts['class_1'] += tf.reduce_sum(class_1)

        return counts

    counts = creditcard_ds.take(10).reduce(
        initial_state={'class_0': 0, 'class_1': 0},
        reduce_func = count
    )

    counts = np.array([
        counts['class_0'].numpy(), counts['class_1'].numpy()
    ]).astype(np.float32)

    fractions = counts/counts.sum()
    logger.info(f'counts/counts.sum(): {fractions}\n')

    # Datasets sampling
    negative_ds = (
      creditcard_ds.unbatch().filter(lambda features, label: label==0).repeat()
    )
    positive_ds = (
      creditcard_ds.unbatch().filter(lambda features, label: label==1).repeat()
    )

    for features, label in positive_ds.batch(10).take(1):
        logger.debug(label.numpy())

    # To use tf.data.experimental.sample_from_datasets pass the datasets, 
    # and the weight for each:

    balanced_ds = tf.data.experimental.sample_from_datasets(
        [negative_ds, positive_ds], 
        [0.5, 0.5]
    ).batch(10)

    # Now the dataset produces examples of each class with 50/50 probability:
    logger.info('balanced_ds.take(10):')
    for features, labels in balanced_ds.take(10):
        print(labels.numpy())
    print()

    # Rejection resampling
    __doc__='''
    One problem with the above experimental.sample_from_datasets approach is
    that it needs a separate tf.data.Dataset per class. Using Dataset.filter
    works, but results in all the data being loaded twice.

    The data.experimental.rejection_resample function can be applied to a
    dataset to rebalance it, while only loading it once. Elements will be
    dropped from the dataset to achieve balance.

    data.experimental.rejection_resample takes a class_func argument. This
    class_func is applied to each dataset element, and is used to determine
    which class an example belongs to for the purposes of balancing.
    '''
    print(__doc__)

    def class_func(features, label):
        return label

    # The resampler also needs a target distribution, and optionally an initial
    # distribution estimate:
    resampler = tf.data.experimental.rejection_resample(
        class_func, 
        target_dist=[0.5, 0.5], 
        initial_dist=fractions
    )

    # The resampler deals with individual examples, so you must unbatch the
    # dataset before applying the resampler:
    resample_ds = creditcard_ds.unbatch().apply(resampler).batch(10)

    # The resampler returns creates (class, example) pairs from the output of
    # the class_func. In this case, the example was already a (feature, label)
    # pair, so use map to drop the extra copy of the labels:
    balanced_ds = resample_ds.map(
        lambda extra_label, features_and_label: features_and_label
    )
    
    # Now the dataset produces examples of each class with 50/50 probability:
    logger.info('balanced_ds.take(10):')
    for features, labels in balanced_ds.take(10):
        print(labels.numpy())


args.step = auto_increment(args.step, args.all)
### Step #20 - Iterator Checkpointing 
if args.step == 20:
    print("\n### Step #20 - Iterator Checkpointing")
    __doc__='''
    Tensorflow supports taking checkpoints so that when your training process
    restarts it can restore the latest checkpoint to recover most of its
    progress. In addition to checkpointing the model variables, you can also
    checkpoint the progress of the dataset iterator. This could be useful if
    you have a large dataset and don't want to start the dataset from the
    beginning on each restart. Note however that iterator checkpoints may be
    large, since transformations such as shuffle and prefetch require buffering
    elements within the iterator.
    '''
    print(__doc__)

    # To include your iterator in a checkpoint, pass the iterator to the
    # tf.train.Checkpoint constructor.
    range_ds = tf.data.Dataset.range(20)

    iterator = iter(range_ds)
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), iterator=iterator)
    manager = tf.train.CheckpointManager(ckpt, '/tmp/tf2_g0401/my_ckpt', max_to_keep=3)

    print([next(iterator).numpy() for _ in range(5)])

    save_path = manager.save()

    print([next(iterator).numpy() for _ in range(5)])

    ckpt.restore(manager.latest_checkpoint)

    print([next(iterator).numpy() for _ in range(5)])


args.step = auto_increment(args.step, args.all)
### Step #21 - Using tf.data with tf.keras
if args.step == 21:
    print("\n### Step #21 - Using tf.data with tf.keras")

    train, test = tf.keras.datasets.fashion_mnist.load_data()

    images, labels = train
    images = images/255.0
    labels = labels.astype(np.int32)

    fmnist_train_ds = tf.data.Dataset.from_tensor_slices((images, labels))
    fmnist_train_ds = fmnist_train_ds.shuffle(5000).batch(32)

    model = Sequential([
        Flatten(),
        Dense(10)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    logger.info('model.fit(fmist_train_ds):')
    model.fit(fmnist_train_ds, epochs=2, verbose=2)
    print()

    logger.info('model.fit(fmist_train_ds.repeat()):')
    model.fit(fmnist_train_ds.repeat(), epochs=2, steps_per_epoch=20, verbose=2)
    print()

    # For evaluation you can pass the number of evaluation steps:
    loss, accuracy = model.evaluate(fmnist_train_ds, verbose=0)
    logger.info(f"Loss : {loss:.2f}")
    logger.info(f"Accuracy : {accuracy:.2f}\n")

    # For long datasets, set the number of steps to evaluate:
    loss, accuracy = model.evaluate(fmnist_train_ds.repeat(), steps=10, verbose=0)
    logger.info(f"Loss : {loss:.2f}")
    logger.info(f"Accuracy : {accuracy:.2f}\n")

    # The labels are not required in when calling Model.predict.
    predict_ds = tf.data.Dataset.from_tensor_slices(images).batch(32)
    result = model.predict(predict_ds, steps = 10) # 32 batch x 10 steps
    logger.info(f'result.shape: {result.shape}')  # (320, 10)

    # But the labels are ignored if you do pass a dataset containing them:
    result = model.predict(fmnist_train_ds, steps = 10)
    logger.info(f'result.shape: {result.shape}')  # (320, 10)


### End of File
print()
if args.plot:
    plt.show()
debug()

