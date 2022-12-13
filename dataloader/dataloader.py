from typing import Tuple
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds


def _normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label # type: ignore


def load_mnist_dataset(batch_size: int=128, batch: bool=True) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
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
