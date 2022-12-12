import argparse

import tensorflow as tf

from dataloader.dataloader import load_mnist_dataset
from model.hparams import read_config
from preprocessing.data_augmentation import get_data_augmentation_layer
from plot.learning_curve import plot_learning_curve


def get_cnn_model() -> tf.keras.Model:
    model = tf.keras.models.Sequential([
        get_data_augmentation_layer(image_size=28, normalization=False),
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
        ])
    return model


def _run_experiment(model, ds_train: tf.data.Dataset, ds_test: tf.data.Dataset) -> tf.keras.callbacks.History:
    # --- Read config ---
    config = read_config()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    history = model.fit(
        ds_train,
        epochs=config["num_epochs"],
        validation_data=ds_test,
    )
    
    return history


def main():
    ds_train, ds_test = load_mnist_dataset()
    model = get_cnn_model()
    history = _run_experiment(model, ds_train, ds_test)
    plot_learning_curve(history=history, filepath="figs/learning_curve_mnist-CNN.png")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vision Transformer Classifier for MNIST.")
    main()
