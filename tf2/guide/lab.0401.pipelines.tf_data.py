#!/usr/bin/env python

import sys
sys.path.append('./')
sys.path.append('../../')

from lab_utils import (
    tf, os, np, plt, logger, ap, BooleanAction,
    debug, toc, auto_increment
)

ap.add_argument('--epochs', type=int, default=2, help='number of epochs: 2*')
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

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Layer, Dense


args.step = auto_increment(args.step, args.all)
### Step #0: Donwload flower photos
if args.step:
    print("\n### Step #0: Donwload flower photos")

    flowers_root = tf.keras.utils.get_file(
        'flower_photos',
        'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
        untar=True
    )


args.step = auto_increment(args.step, args.all)
### Step #1 - Basic mechanics
if args.step == 1:
    print("\n### Step #1 - Basic mechanics")

    dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])
    logger.info(dataset)

    for elem in dataset:  
        logger.info(elem.numpy())

    it = iter(dataset)
    logger.info(next(it).numpy())

    logger.info(dataset.reduce(0, lambda state, value: state + value).numpy())

    dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))
    logger.info(dataset1.element_spec)
    for element in dataset1:
        logger.info(element.numpy())

    dataset2 = tf.data.Dataset.from_tensor_slices( 
        (tf.random.uniform([4]), 
         tf.random.uniform([4, 100], maxval=100, dtype=tf.int32))
    )
    logger.info(dataset2.element_spec)

    dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
    logger.info(dataset3.element_spec)
    for a, (b,c) in dataset3:
        logger.info('shapes: {a.shape}, {b.shape}, {c.shape}'.format(a=a, b=b, c=c))

    # Dataset containing a sparse tensor.
    sparse_tensor = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
    logger.info(f'\n{tf.sparse.to_dense(sparse_tensor)}')
    dataset4 = tf.data.Dataset.from_tensors(sparse_tensor)
    logger.info(dataset4.element_spec)


args.step = auto_increment(args.step, args.all)
### Step #2 - Reading input data: Consuming NumPy arrays
if args.step == 2:
    print("\n### Step #2 - Reading input data: Consuming NumPy arrays")

    train, test = tf.keras.datasets.fashion_mnist.load_data()
    images, labels = train
    images = images/255

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    for image, label in dataset.take(1):
        logger.info(f"image's shape: {image.shape}, label's shape: {label.shape}")
    for image_batch, label_batch in dataset.batch(32).take(1):
        logger.info(f"image_batch's shape: {image_batch.shape}, label_batch's shape: {label_batch.shape}")


args.step = auto_increment(args.step, args.all)
### Step #3 - Reading input data: Consuming Python generators
if args.step == 3:
    print("\n### Step #3 - Reading input data: Consuming Python generators")

    def count(stop):
        i = 0
        while i<stop:
            yield i
            i += 1

    for n in count(5):
        logger.debug(n)

    ds_counter = tf.data.Dataset.from_generator(
        count, args=[25], 
        output_types=tf.int32, 
        output_shapes = (), 
    )

    for count_batch in ds_counter.batch(10).take(3):
        logger.info(count_batch.numpy())

    for count_batch in ds_counter.repeat(3).batch(10).take(3):
        logger.info(count_batch.numpy())

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

    ds_series_batch = ds_series.shuffle(20).padded_batch(10)
    ids, sequence_batch = next(iter(ds_series_batch))
    logger.info(f'{ids.numpy()}')
    logger.info(f'\n{sequence_batch.numpy()}')

    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20)

    images, labels = next(img_gen.flow_from_directory(flowers_root, batch_size=32))
    logger.debug(f'{images.dtype}, {images.shape}')
    logger.debug(f'{labels.dtype}, {labels.shape}')

    ds = tf.data.Dataset.from_generator(
        lambda: img_gen.flow_from_directory(flowers_root), 
        output_types=(tf.float32, tf.float32), 
        output_shapes=([32,256,256,3], [32,5])
    )
    logger.info(ds.element_spec)

    for images, label in ds.take(1):
        logger.info('images.shape: {}'.format(images.shape))
        logger.info('labels.shape: {}'.format(labels.shape))


args.step = auto_increment(args.step, args.all)
### Step #4 - Reading input data: Consuming TFRecord data
if args.step == 4:
    print("\n### Step #4 - Reading input data: Consuming TFRecord data")

    # Creates a dataset that reads all of the examples from two files.
    fsns_test_file = tf.keras.utils.get_file("fsns.tfrec", "https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001")
    dataset = tf.data.TFRecordDataset(filenames = [fsns_test_file])
    logger.info(dataset)
    
    raw_example = next(iter(dataset))
    parsed = tf.train.Example.FromString(raw_example.numpy())

    logger.info(parsed.features.feature['image/text'])


args.step = auto_increment(args.step, args.all)
### Step #5 - Reading input data: Consuming text data
if args.step == 5:
    print("\n### Step #5 - Reading input data: Consuming text data")

    directory_url = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
    file_names = ['cowper.txt', 'derby.txt', 'butler.txt']

    file_paths = [
        tf.keras.utils.get_file(file_name, directory_url + file_name)
        for file_name in file_names
    ]

    dataset = tf.data.TextLineDataset(file_paths)
    for line in dataset.take(5):
        logger.info(line.numpy())

    #
    files_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    lines_ds = files_ds.interleave(tf.data.TextLineDataset, cycle_length=3)

    for i, line in enumerate(lines_ds.take(9)):
      if i % 3 == 0:
        tf.print()
      logger.info(line.numpy())

    #
    titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
    titanic_lines = tf.data.TextLineDataset(titanic_file)
    for line in titanic_lines.take(5):
        logger.info(line.numpy())

    # 0th 위치, 1길이 값이 "0"이 아닌 라인
    def survived(line):
        return tf.not_equal(tf.strings.substr(line, 0, 1), "0") 

    survivors = titanic_lines.skip(1).filter(survived) # remove the header line
    for line in survivors.take(5):
        logger.info(line.numpy())
    

args.step = auto_increment(args.step, args.all)
### Step #6 - Reading input data: Consuming CSV data
if args.step == 6:
    print("\n### Step #6 - Reading input data: Consuming CSV data")

    titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
    df = pd.read_csv(titanic_file)
    logger.info(f'\n{df.head()}')

    # in-memory
    tf.print("\n# In-momory")
    titanic_slices = tf.data.Dataset.from_tensor_slices(dict(df))
    for feature_batch in titanic_slices.take(1):
        for key, value in feature_batch.items():
            logger.info("{!r:20s}: {}".format(key, value))

    # from disk
    tf.print("\n# from disk")
    titanic_batches = tf.data.experimental.make_csv_dataset(
        titanic_file, 
        batch_size=4,
        label_name="survived"
    )

    for feature_batch, label_batch in titanic_batches.take(1):
        logger.info("survived: {}".format(label_batch))
        logger.info("features:")
        for key, value in feature_batch.items():
            logger.info("{!r:20s}: {}".format(key, value))

    # select columns
    tf.print("\n# from disk: select columns")
    titanic_batches = tf.data.experimental.make_csv_dataset(
        titanic_file, 
        batch_size=4,
        label_name="survived", 
        select_columns=['class', 'fare', 'survived']
    )

    for feature_batch, label_batch in titanic_batches.take(1):
        logger.info("survived: {}".format(label_batch))
        logger.info("features:")
        for key, value in feature_batch.items():
            logger.info("{!r:20s}: {}".format(key, value))
    
    # lower-level
    tf.print("\n# lower-level")
    titanic_types  = [tf.int32, tf.string, tf.float32, tf.int32, tf.int32, tf.float32, tf.string, tf.string, tf.string, tf.string]
    dataset = tf.data.experimental.CsvDataset(titanic_file, titanic_types , header=True)

    for line in dataset.take(10): 
        logger.info([item.numpy() for item in line])


args.step = auto_increment(args.step, args.all)
### Step #7 - Reading input data: Consuming sets of files
if args.step == 7:
    print("\n### Step #7 - Reading input data: Consuming sets of files")

    flowers_root = pathlib.Path(flowers_root)
    list_ds = tf.data.Dataset.list_files(str(flowers_root/'*/*'))
    for f in list_ds.take(5):
        logger.info(f.numpy())

    def process_path(file_path):
        label = tf.strings.split(file_path, os.sep)[-2]
        return tf.io.read_file(file_path), label

    labeled_ds = list_ds.map(process_path)
    for image_raw, label_text in labeled_ds.take(1):
        logger.info(repr(image_raw.numpy()[:100]))
        tf.print()
        logger.info(label_text.numpy())


args.step = auto_increment(args.step, args.all)
### Step #8 - Batching dataset elements: Simple batching
if args.step == 8:
    print("\n### Step #8 - Batching dataset elements: Simple batching")

    inc_dataset = tf.data.Dataset.range(20)
    dec_dataset = tf.data.Dataset.range(0, -20, -1)
    dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
    batched_dataset = dataset.batch(4)

    for batch in batched_dataset.take(4):
        logger.info([arr.numpy() for arr in batch])

    tf.print("\n# using drop_remainder")
    batched_dataset = dataset.batch(7, drop_remainder=True)
    for batch in batched_dataset:
        logger.info([arr.numpy() for arr in batch])


args.step = auto_increment(args.step, args.all)
### Step #9 - Batching dataset elements: Batching tensors with padding
if args.step == 9:
    print("\n### Step #9 - Batching dataset elements: Batching tensors with padding")

    dataset = tf.data.Dataset.range(100)
    dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
    dataset = dataset.padded_batch(4, padded_shapes=(None,))

    for i, batch in enumerate(dataset.take(3)):
        logger.info(f'{i+1}th batch:\n{batch.numpy()}')
        tf.print()


args.step = auto_increment(args.step, args.all)
### Step #10 - Training workflows: Processing multiple epochs
if args.step == 10:
    print("\n### Step #10 - Training workflows: Processing multiple epochs")

    titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
    titanic_lines = tf.data.TextLineDataset(titanic_file)

    def plot_batch_sizes(ds):
        batch_sizes = [batch.shape[0] for batch in ds]
        plt.figure()
        plt.bar(range(len(batch_sizes)), batch_sizes)
        plt.xlabel('Batch number')
        plt.ylabel('Batch size')
        plt.show(block=False)

    if args.plot:
        titanic_batches = titanic_lines.repeat(3).batch(128)
        plot_batch_sizes(titanic_batches)

        titanic_batches = titanic_lines.batch(128).repeat(3)
        plot_batch_sizes(titanic_batches)

    epochs = 3
    dataset = titanic_lines.batch(128)

    for epoch in range(epochs):
        for batch in dataset:
            logger.info(batch.shape)
        logger.info("End of epoch: %d" % epoch)


args.step = auto_increment(args.step, args.all)
### Step #11 - Training workflows: Randomly shuffling input data
if args.step == 11:
    tf.print("\n### Step #11 - Training workflows: Randomly shuffling input data")

    titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
    lines = tf.data.TextLineDataset(titanic_file)
    counter = tf.data.experimental.Counter()

    dataset = tf.data.Dataset.zip((counter, lines))
    dataset = dataset.shuffle(buffer_size=30)
    dataset = dataset.batch(10)
    for n, line_batch in dataset.take(3):
        logger.info(n.numpy()) 

    #
    dataset = tf.data.Dataset.zip((counter, lines))
    shuffled = dataset.shuffle(buffer_size=100).batch(10).repeat(2)

    tf.print("\nHere are the item ID's near the epoch boundary:")
    for n, line_batch in shuffled.skip(60).take(5):
        logger.info(n.numpy()) 

    dataset = tf.data.Dataset.zip((counter, lines))
    shuffled = dataset.repeat(2).shuffle(buffer_size=100).batch(10)

    tf.print("\nHere are the item ID's near the epoch boundary:\n")
    for n, line_batch in shuffled.skip(55).take(15):
        logger.info(n.numpy()) 

    if args.plot:
        shuffle_repeat = [n.numpy().mean() for n, line_batch in shuffled]
        repeat_shuffle = [n.numpy().mean() for n, line_batch in shuffled]
        plt.plot(shuffle_repeat, label="shuffle().repeat()")
        plt.plot(repeat_shuffle, label="repeat().shuffle()")
        plt.ylabel("Mean item ID")
        plt.legend()


args.step = auto_increment(args.step, args.all)
### Step #12 - Preprocessing data: Decoding image data and resizing it
if args.step in [12, 13]:
    print("\n### Step #12 - Preprocessing data: Decoding image data and resizing it")
    
    flowers_root = pathlib.Path(flowers_root)
    list_ds = tf.data.Dataset.list_files(str(flowers_root/'*/*'))

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

    file_path = next(iter(list_ds))
    image, label = parse_image(file_path)

    def show(image, label):
        plt.figure()
        plt.imshow(image)
        plt.title(label.numpy().decode('utf-8'))
        plt.axis('off')
        plt.show(block=False)
        args.plot = True

    images_ds = list_ds.map(parse_image)

    if args.step == 12:
        for image, label in images_ds.take(5):
            show(image, label)

        
args.step = auto_increment(args.step, args.all)
### Step #13 - Preprocessing data: Applying arbitrary Python logic 
if args.step == 13:
    print("\n### Step #13 - Preprocessing data: Applying arbitrary Python logic")

    import scipy.ndimage as ndimage
    def random_rotate_image(image):
        image = ndimage.rotate(image, np.random.uniform(-30, 30), reshape=False)
        return image

    image, label = next(iter(images_ds))
    image = random_rotate_image(image)
    show(image, label)

    def tf_random_rotate_image(image, label):
        im_shape = image.shape
        [image,] = tf.py_function(random_rotate_image, [image], [tf.float32])
        image.set_shape(im_shape)
        return image, label

    rot_ds = images_ds.map(tf_random_rotate_image)
    for image, label in rot_ds.take(5):
        show(image, label)


args.step = auto_increment(args.step, args.all)
### Step #14 - Preprocessing data: Parsing tf.Example protocol buffer messages
if args.step == 14:
    print("\n### Step #14 - Preprocessing data: Parsing tf.Example protocol buffer messages")

    fsns_test_file = tf.keras.utils.get_file(
        "fsns.tfrec", 
        "https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001"
    )

    dataset = tf.data.TFRecordDataset(filenames = [fsns_test_file])

    raw_example = next(iter(dataset))
    parsed = tf.train.Example.FromString(raw_example.numpy())

    feature = parsed.features.feature
    raw_img = feature['image/encoded'].bytes_list.value[0]
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
    logger.info(txt.numpy())
    logger.info(repr(img.numpy()[:20]))

    decoded = dataset.map(tf_parse)
    image_batch, text_batch = next(iter(decoded.batch(10)))
    for text in text_batch:
        logger.info(text.numpy())


args.step = auto_increment(args.step, args.all)
### Step #15 - Preprocessing data: Time series windowing
if args.step == 15:
    print("\n### Step #15 - Preprocessing data: Time series windowing")


args.step = auto_increment(args.step, args.all)
### Step #16 - Preprocessing data: Resampling
if args.step == 16:
    print("\n### Step #16 - Preprocessing data: Resampling")


args.step = auto_increment(args.step, args.all)
### Step #17 - Iterator Checkpointing 
if args.step == 17:
    print("\n### Step #17 - Iterator Checkpointing")
    

args.step = auto_increment(args.step, args.all)
### Step #18 - Using tf.data with tf.keras
if args.step == 18:
    print("\n### Step #18 - Using tf.data with tf.keras")





    ### End of File
if args.plot:
    plt.show()
debug()

