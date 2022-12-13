import sys, os
sys.path.append(os.path.dirname(".."))

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tqdm.auto import tqdm

from dataloader.dataloader import load_mnist_dataset
from model.hparams import read_config
from model.vision_transformer import create_vit_classifier
from train_CNN import get_cnn_model

import matplotlib.pyplot as plt


# LIST_N_EXAMPLES_TRAIN = list(range(10000, 60001, 10000))
LIST_N_EXAMPLES_TRAIN = [1000]
FIG_DIRPATH = Path("figs/experiments")
FIG_DIRPATH.mkdir(parents=True, exist_ok=True)


def _plot_and_save_history(history_per_n_examples: Dict[int, List[int]], filepath: str,
                           title: Optional[str]=None):
    fig, ax = plt.subplots()
    df = pd.DataFrame(history_per_n_examples)
    if title is None:
        title = "Learning curve"
    df.plot(ax=ax, xlabel="Epochs", ylabel="Validation loss (cross-entropy)", title=title)
    fig.legend(title="Number of examples in the train set")
    fig.tight_layout()
    fig.savefig(filepath)
    return


def main():
    # --- Read config ---
    config = read_config()
    
    
    # --- Get data ---
    # (x_train, y_train), (x_test, y_test) = load_mnist()
    ds_train, ds_test = load_mnist_dataset(batch=False)
    ds_test = ds_test.batch(config["batch_size"]).cache().prefetch(tf.data.AUTOTUNE)
    
    
    # --- Get CNN model ---
    cnn_model = get_cnn_model()
    
    # --- Get VIT model ---
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
    
    
    # --- Compile models ---
    optimizer = tf.optimizers.Adam(learning_rate=config["learning_rate"])
    vit_classifier.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    
    cnn_model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    
    
    # --- Training ---
    history_vit = {}
    history_cnn = {}
    
    for n_examples_train in tqdm(LIST_N_EXAMPLES_TRAIN):
        ds_train_subset = ds_train.take(n_examples_train).batch(config["batch_size"]).prefetch(tf.data.AUTOTUNE)
        
        history_vit[n_examples_train] = vit_classifier.fit(
            ds_train_subset,
            epochs=config["num_epochs"],
            validation_data=ds_test
        ).history["val_loss"]
        
        history_cnn[n_examples_train] = cnn_model.fit(
            ds_train_subset,
            epochs=config["num_epochs"],
            validation_data=ds_test,
        ).history["val_loss"]
    
    
    # --- Plot ---
    _plot_and_save_history(history_vit, filepath=FIG_DIRPATH/"learning_curve_wrt_ds_size-VIT.png", # type: ignore
                           title="Impact of the dataset size on how the VIT learning")
    _plot_and_save_history(history_cnn, filepath=FIG_DIRPATH/"learning_curve_wrt_ds_size-CNN.png", # type: ignore
                           title="Impact of the dataset size on how the CNN learning")
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate figures to analyze the impact of the dataset size on the learning")
    main()    
