import numpy as np
import tensorflow as tf

from generate import GENERATE_TOKEN
from plogging import logger


def vote(X, y_pred, verbose=False):
    """
    return the predicted word based on voting between words
    """
    X = X[X.img_name != GENERATE_TOKEN]  # generated image do not vote
    X["pred"] = np.argmax(y_pred, axis=1)
    grouped = X.groupby(by=["img_name", "word"])
    for name, group in grouped:
        img_name, word = name
        counts = group["pred"].value_counts()
        best = counts.idxmax()
        if verbose:
            logger.info(f"group {name} voted on {best}, group_len={len(group)}")
        X.loc[(X.img_name == img_name) & (X.word == word), "pred"] = best
    return tf.keras.utils.to_categorical(X["pred"])


def votev2(X, y_pred, verbose=False):
    """
    same as v1, only each char votes, and then each word, in reality, I tested both and it didn't really make such of a difference
    """
    X = X[X.img_name != GENERATE_TOKEN]  # generated image do not vote
    X["pred"] = np.argmax(y_pred, axis=1)
    grouped = X.groupby(by=["img_name", "word", "char"])

    # first vote, per char in word
    for name, group in grouped:
        img_name, word, char = name
        counts = group["pred"].value_counts()
        best = counts.idxmax()
        X.loc[
            (X.img_name == img_name) & (X.word == word) & (X.char == char), "pred"
        ] = best

    # second vote, per char per word
    grouped = X.groupby(by=["img_name", "word"])
    for name, group in grouped:
        img_name, word = name
        counts = group["pred"].value_counts()
        best = counts.idxmax()
        if verbose:
            logger.info(f"group {name} voted on {best}, group_len={len(group)}")
        X.loc[(X.img_name == img_name) & (X.word == word), "pred"] = best
    return tf.keras.utils.to_categorical(X["pred"])
