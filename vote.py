import numpy as np
import tensorflow as tf

from plogging import logger


def vote(X, y_pred, verbose=False):
    """
    return the predicted word based on voting between words
    """
    X["pred"] = np.argmax(y_pred, axis=1).astype(int)
    grouped = X.groupby(by=["img_name", "word"])
    for name, group in grouped:
        img_name, word = name
        counts = group["pred"].value_counts()
        best = counts.idxmax()
        if verbose:
            logger.info(f"group {name} voted on {best}, group_len={len(group)}")
        X.loc[(X.img_name == img_name) & (X.word == word), "pred"] = best
    return tf.keras.utils.to_categorical(X["pred"])
