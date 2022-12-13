import tensorflow as tf

from model.hparams import read_config
from preprocessing.data_augmentation import get_data_augmentation_layer


def get_cnn_model() -> tf.keras.Model:
    config = read_config()
    
    model = tf.keras.models.Sequential([
        get_data_augmentation_layer(image_size=config["image_size"], normalization=False),
        tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu', input_shape=config["input_shape"]),
        tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(strides=(2,2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(config["num_classes"], activation=None)
    ])
    
    return model
