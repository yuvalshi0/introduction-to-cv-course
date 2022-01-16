import itertools

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import auc, roc_curve

from config import config

CLASSES = config.get_classes()


def plot_confusion_matrix(
    y_test, y_pred, title="Confusion matrix", cmap=plt.cm.Blues, save=False
):
    """
    Taken from: https://deeplizard.com/learn/video/km7pxKy4UHU with some adjustions
    This function prints and plots the confusion matrix, normalized
    """
    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    cm = tf.math.confusion_matrix(y_test, y_pred).numpy()

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(CLASSES))
    plt.xticks(tick_marks, CLASSES, rotation=45)
    plt.yticks(tick_marks, CLASSES)

    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    if save:
        plt.savefig("metrics/cmatrix.png")
    plt.show()


def plot_acc(history, save=False):

    # summarize history for accuracy
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    if save:
        plt.savefig("metrics/accuracy.png")
    plt.show()


def plot_loss(history, save=False):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    if save:
        plt.savefig("metrics/loss.png")
    plt.show()


def plot_roc(y_test, y_pred, zoom=True, save=False):
    """
    taken from: https://gist.github.com/Tony607/82f7dad24fc122a78d1bdd69e76fbffe with small adjustments
    """
    n_classes = len(CLASSES)
    lw = 2
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    thresholds = {}
    for i in range(n_classes):
        fpr[CLASSES[i]], tpr[CLASSES[i]], thresholds[CLASSES[i]] = roc_curve(
            y_test[:, i], y_pred[:, i], drop_intermediate=False
        )
        roc_auc[CLASSES[i]] = auc(fpr[CLASSES[i]], tpr[CLASSES[i]])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[klass] for klass in CLASSES]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for c in CLASSES:
        mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(1)
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})" "".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})" "".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = itertools.cycle(mcolors.TABLEAU_COLORS.keys())
    for c, color in zip(CLASSES, colors):
        plt.plot(
            fpr[c],
            tpr[c],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})" "".format(c, roc_auc[c]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of Receiver operating characteristic to multi-class")
    plt.legend(loc="lower right")
    if save:
        plt.savefig("metrics/roc.png")
    plt.show()

    # Zoom in view of the upper left corner.
    if zoom:
        plt.figure(2)
        plt.xlim(0, 0.4)
        plt.ylim(0.6, 1)
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average ROC curve (area = {0:0.2f})"
            "".format(roc_auc["micro"]),
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label="macro-average ROC curve (area = {0:0.2f})"
            "".format(roc_auc["macro"]),
            color="navy",
            linestyle=":",
            linewidth=4,
        )

        colors = itertools.cycle(mcolors.TABLEAU_COLORS.keys())
        for c, color in zip(CLASSES, colors):
            plt.plot(
                fpr[c],
                tpr[c],
                color=color,
                lw=lw,
                label="ROC curve of class {0} (area = {1:0.2f})"
                "".format(c, roc_auc[c]),
            )
    plt.show()


def to_csv(x_test, y, csv_file="results.csv"):
    df = x_test.copy()
    df["pred"] = np.argmax(y, axis=1)

    for klass in CLASSES:
        x_test[klass] = 0  # add classes column
    df["pred"] = df["pred"].apply(lambda idx: CLASSES[idx])  # back to label
    for index, row in df.iterrows():
        df.at[index, row.pred] = 1

    df = df.drop(columns=["pred", "img"], errors="ignore")
    df = df.rename(columns={"img_name": "img"})
    df.to_csv(csv_file)


def log_stats(y_test, y_pred, save=True, file_path="metrics/stats.txt"):
    recall = tf.keras.metrics.Recall()
    recall.update_state(y_test, y_pred)
    precision = tf.keras.metrics.Precision()
    precision.update_state(y_test, y_pred)
    auc = tf.keras.metrics.AUC()
    auc.update_state(y_test, y_pred)
    acc = tf.keras.metrics.CategoricalAccuracy()
    acc.update_state(y_test, y_pred)

    print(f"Accuracy: {acc.result().numpy()}")
    print(f"Recall: {recall.result().numpy()}")
    print(f"Precision: {precision.result().numpy()}")
    print(f"AUC: {auc.result().numpy()}")

    if save:
        with open(file_path, "w") as f:
            print(f"Accuracy: {acc.result().numpy()}", file=f)
            print(f"Recall: {recall.result().numpy()}", file=f)
            print(f"Precision: {precision.result().numpy()}", file=f)
            print(f"AUC: {auc.result().numpy()}", file=f)
