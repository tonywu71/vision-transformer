from typing import Tuple
import numpy as np

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import tensorflow_datasets as tfds


def load_mnist(normalization: bool=True, one_hot: bool=False) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                                                       Tuple[np.ndarray, np.ndarray]]:
    # Loading pre-shuffled MNIST data
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Reshaping images represented as 2 dimensional (N, w*h) into 4 dimensional arrays (N, w, h, 1)
    # gray scale intensity values to reside in an interval [0,1]
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

    if normalization:
        train_images = train_images / 255.
        test_images = test_images / 255.
    
    if one_hot:
        # Converting digit class id's into a one-hot encoding
        train_labels = to_categorical(train_labels, 10)
        test_labels = to_categorical(test_labels, 10)

    return (train_images, train_labels), (test_images, test_labels)


def _normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label # type: ignore


def load_mnist_dataset(batch: bool=True) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.map(
        _normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    if batch:
        ds_train = ds_train.batch(128)
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)


    ds_test = ds_test.map(
        _normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    if batch:
        ds_test = ds_test.batch(128)
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    
    return ds_train, ds_test
