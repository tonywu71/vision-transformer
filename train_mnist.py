import argparse
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

from dataloader.dataloader import load_mnist
from model.hparams import read_config
from model.vision_transformer import create_vit_classifier
from plot.learning_curve import plot_learning_curve


def run_experiment(model, x_train, y_train, x_test, y_test) -> tf.keras.callbacks.History:
    # --- Read config ---
    config = read_config()
    
    optimizer = tf.optimizers.Adam(learning_rate=config["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )


    # --- CHECKPOINTS ---
    checkpoint_filepath = "./checkpoints/mnist"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        patience=3,
        monitor="val_loss",
        mode="min",
        restore_best_weights=True
    )
    
    log_dir = f'logs/mnist/' + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    
    # --- TRAINING ---
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=config["batch_size"],
        epochs=config["num_epochs"],
        validation_split=0.2,
        callbacks=[checkpoint_callback, early_stopping_callback, tensorboard_callback],
    )


    # --- EVALUATION ---
    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history


def main():
    # --- Read config ---
    config = read_config()
    
    # --- Prepare the data ---
    (x_train, y_train), (x_test, y_test) = load_mnist()

    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")


    # --- Get model ---
    vit_classifier = create_vit_classifier(input_shape=config["input_shape"],
                                           num_classes=config["num_classes"],
                                           image_size=config["image_size"],
                                           patch_size=config["patch_size"],
                                           num_patches=config["num_patches"],
                                           projection_dim=config["projection_dim"],
                                           dropout=config["dropout"],
                                           n_transformer_layers=config["n_transformer_layers"],
                                           num_heads=config["num_heads"],
                                           transformer_units=config["transformer_units"],
                                           mlp_head_units=config["mlp_head_units"])

    # --- Training ---
    history = run_experiment(vit_classifier, x_train, y_train, x_test, y_test)
    plot_learning_curve(history=history, filepath="figs/learning_curve_mnist-VIT.png")
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vision Transformer Classifier for MNIST.")
    main()    
