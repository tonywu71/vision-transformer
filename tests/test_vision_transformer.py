import tensorflow as tf
from tensorflow import keras

from model.hparams import read_config
from model.vision_transformer import create_vit_classifier



def _compile_vit(model):
    # --- Read config ---
    config = read_config()
    
    optimizer = tf.optimizers.Adam(learning_rate=config["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ]
    )
    
    return


def test_create_vit_training_on_subsample_of_mnist():
    # --- Read config ---
    config = read_config()
    
    
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
    _compile_vit(vit_classifier)
    
    return
