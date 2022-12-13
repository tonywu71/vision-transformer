import logging

import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)


def plot_learning_curve(history: tf.keras.callbacks.History, filepath: str):
    history_loss = pd.DataFrame(history.history, columns=["loss", "val_loss"])
    ax = history_loss.plot()
    fig = ax.get_figure() # type: ignore
    fig.savefig(filepath)
    
    logger.info("Successfully generated and saved training history figures.")
    return
