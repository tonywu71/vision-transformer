import argparse
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

from dataloader.dataloader import load_mnist
from model.vision_transformer import create_vit_classifier
from plot.learning_curve import plot_learning_curve


# --- Hyperparameters ---
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 1
image_size = 28  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
n_transformer_layers = 8
mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier



def run_experiment(model, x_train, y_train, x_test, y_test) -> tf.keras.callbacks.History:
    # optimizer = tfa.optimizers.AdamW(
    #     learning_rate=learning_rate, weight_decay=weight_decay
    # )
    
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "/checkpoints/mnist"
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
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=5)

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback, early_stopping_callback, tensorboard_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history


def main():
    # --- Prepare the data ---
    num_classes = 10
    input_shape = (28, 28, 1)

    (x_train, y_train), (x_test, y_test) = load_mnist()

    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")


    # --- Get model ---
    vit_classifier = create_vit_classifier(input_shape=input_shape,
                                           num_classes=num_classes,
                                           image_size=image_size,
                                           patch_size=patch_size,
                                           num_patches=num_patches,
                                           projection_dim=projection_dim,
                                           n_transformer_layers=n_transformer_layers,
                                           num_heads=num_heads,
                                           transformer_units=transformer_units,
                                           mlp_head_units=mlp_head_units)

    # --- Training ---
    history = run_experiment(vit_classifier, x_train, y_train, x_test, y_test)
    plot_learning_curve(history=history)
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vision Transformer Classifier for MNIST.")
    main()    
