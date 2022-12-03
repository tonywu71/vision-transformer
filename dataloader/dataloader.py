from typing import Tuple
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist


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
