from typing import List
import tensorflow as tf
from tensorflow.keras import layers


def mlp(x: tf.Tensor, hidden_units: List[int], dropout_rate: float) -> tf.Tensor:
    """Multi-Layer Perceptron

    Args:
        x (tf.Tensor): Input
        hidden_units (List[int])
        dropout_rate (float)

    Returns:
        tf.Tensor: Output
    """
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
