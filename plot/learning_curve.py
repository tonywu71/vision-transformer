import logging

import pandas as pd
import tensorflow as tf

import seaborn as sns

logger = logging.getLogger(__name__)
sns.set_theme()


def plot_learning_curve(history: tf.keras.callbacks.History):
    history_loss = pd.DataFrame(history.history, columns=["loss", "val_loss"])
    ax = history_loss.plot()
    fig = ax.get_figure() # type: ignore
    fig.savefig(f'figs/learning_curve-loss.png')
    
    logger.info("Successfully generated and saved training history figures.")
