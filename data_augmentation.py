import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa


def get_data_augmentation_layer(image_size: int, normalization: bool=True) -> keras.layers.Layer:
    list_layers = []
    
    if normalization:
        list_layers.append(layers.Normalization())
    
    list_layers.extend([
            layers.Resizing(image_size, image_size),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ])
    
    data_augmentation = keras.Sequential(
        list_layers,
        name="data_augmentation",
    )
    return data_augmentation

