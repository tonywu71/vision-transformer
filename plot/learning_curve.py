import logging

import pandas as pd
import tensorflow as tf

import seaborn as sns


logger = logging.getLogger(__name__)
sns.set_theme()


def plot_learning_curve(history: tf.keras.callbacks.History, filepath: str):
    history_loss = pd.DataFrame(history.history, columns=["loss", "val_loss"])
    ax = history_loss.plot(xlabel="Epochs", ylabel="Validation loss (cross-entropy)",
                           title="Learning Curve")
    fig = ax.get_figure() # type: ignore
    fig.savefig(filepath)
    
    logger.info("Successfully generated and saved training history figures.")
    return
