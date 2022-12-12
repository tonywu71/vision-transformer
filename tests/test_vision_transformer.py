import tensorflow as tf
from tensorflow import keras

from dataloader.dataloader import load_mnist
from model.vision_transformer import create_vit_classifier
from plot.learning_curve import plot_learning_curve


# --- Hyperparameters ---
learning_rate = 1e-4
weight_decay = 0.0001
batch_size = 64
num_epochs = 1 # TEST ONLY
image_size = 28  # We'll resize input images to this size
patch_size = 7  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 256
dropout = 0.2
num_heads = 8
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
n_transformer_layers = 3
mlp_head_units = [256]  # Size of the dense layers of the final classifier



def run_experiment(model, x_train, y_train, x_test, y_test) -> tf.keras.callbacks.History:
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1
    )

    return history


def test_vit_training_on_subsample_of_mnist():
    # --- Prepare the data ---
    num_classes = 10
    input_shape = (28, 28, 1)
    N_EXAMPLES_TRAIN = 1000
    N_EXAMPLES_TEST = 200

    (x_train, y_train), (x_test, y_test) = load_mnist()
    
    
    # --- SUBSAMPLE DATASET FOR TESTING PURPOSES ---
    x_train = x_train[:N_EXAMPLES_TRAIN, :]
    y_train = x_train[:N_EXAMPLES_TRAIN]
    x_test = x_train[:N_EXAMPLES_TEST, :]
    x_test = x_train[:N_EXAMPLES_TEST]
    

    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")


    # --- Get model ---
    vit_classifier = create_vit_classifier(input_shape=input_shape,
                                           num_classes=num_classes,
                                           image_size=image_size,
                                           patch_size=patch_size,
                                           num_patches=num_patches,
                                           projection_dim=projection_dim,
                                           dropout=dropout,
                                           n_transformer_layers=n_transformer_layers,
                                           num_heads=num_heads,
                                           transformer_units=transformer_units,
                                           mlp_head_units=mlp_head_units)

    # --- Training ---
    history = run_experiment(vit_classifier, x_train, y_train, x_test, y_test)
    plot_learning_curve(history=history, filepath="tests/learning_curve_mnist_TEST.png")
    
    return
