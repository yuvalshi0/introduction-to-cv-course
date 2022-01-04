import os

from config import config

CODA_DLL_PATH = config["preprocessing"]["coda_dll_path"]
os.add_dll_directory(
    CODA_DLL_PATH
)  # https://github.com/tensorflow/tensorflow/issues/48868#issuecomment-841396124


import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from augment import BAD_augment_image, augment_image, augment_image_v2
from plogging import log_time, logger

IMG_SIZE = int(config["main"]["img_size"])
CLASSES = config.get_classes()


def _limit_correction(pts, shape):
    """
    fix the limits of the bounding box, as they can be out of the picture
    """
    heigh, width, _ = shape

    def fix(pt):
        x = max(min(width, pt[0]), 0)
        y = max(min(heigh, pt[1]), 0)
        return x, y

    return np.apply_along_axis(fix, 1, pts)


def _mk_points(bb):
    return np.array(list((zip(bb[0], bb[1]))), dtype="int")


def crop(img, bb):
    """
    Cuts the image by the bounding box, also applies a mask by setting the mean as the background
    """
    pts = _mk_points(bb)
    pts = _limit_correction(pts, img.shape)
    x, y, w, h = cv2.boundingRect(pts)
    croped = img[y : y + h, x : x + w]

    # https://stackoverflow.com/a/48301735
    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    dst = cv2.bitwise_and(croped, croped, mask=mask)

    bg = np.ones_like(croped, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    img_ = bg + dst

    # in all pixels where the we have white (like the mask we applied) and the bg also had white -> we place the mean color of the croped image
    for i in range(len(img_)):
        for j in range(len(img_[i])):
            if (img_[i][j] == (255, 255, 255)).all() and (
                bg[i][j] == (255, 255, 255)
            ).all():
                img_[i][j] = np.mean(croped, axis=(0, 1))
    return img_


def rotate(img, bb):
    """
    rotates the image to
    """
    pts = _mk_points(bb)
    """
    pt[0]      pt[1]
        |     |
        ######
        #    #   
        #    #   
        #    #   
        ######
        |    |
    pt[2]     pt[3]

    1. We take the line between pts[2] and pts[3]
    2. Calculate the angle between the line and the main 
    3. Rotate the image with apply_affine_transform and nearest mask 

    """
    angle = -np.rad2deg(np.arctan2(pts[2][1] - pts[3][1], pts[2][0] - pts[3][0]))

    if pts[0][1] > pts[3][1]:  # letter is upside down
        return tf.keras.preprocessing.image.apply_affine_transform(
            img, theta=angle + 180, fill_mode="nearest", shear=180
        )
    return tf.keras.preprocessing.image.apply_affine_transform(
        img, theta=angle, fill_mode="nearest"
    )


def standardize(img):
    """
    resize to 32x32 and make it gray
    """
    img_ = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)


def mk_charlst(lst):
    """
    helper function, returns a list of tuples, where the first element is the char and the second element is the word
    """
    clst = []
    for word in lst:
        word_ = word.decode("utf-8")
        clst += [(c, word_) for c in word_]
    return clst


def _split_data_set(dataset):
    X = dataset.drop(columns=["font"])
    Y = dataset["font"].apply(lambda s: CLASSES.index(s))

    CAT_CLASSES = tf.keras.utils.to_categorical(np.unique(Y))

    f_ = lambda i: CAT_CLASSES[i]

    Y = f_(Y)
    return train_test_split(X, Y, random_state=0, test_size=0.3)


def _augment_dataset(X, Y, augment_method, augment_cycles, verbose=False):
    X["font"] = np.array(Y)  # temporary unite the datasets
    records = X.to_dict("records")
    aug_train = []
    c = 0
    for record in records:
        for _ in range(augment_cycles):
            tmpimg_ = augment_method(record["img"])
            aug_train.append(
                {
                    "img": tmpimg_,
                    "font": record["font"],
                    "char": record["char"],
                    "word": record["word"],
                    "img_name": record["img_name"],
                }
            )
            if verbose:
                c += 1
                img_name = record["img_name"]
                logger.info(
                    f"Finished augmenting image {c}/{len(X)*augment_cycles} [image={img_name}]"
                )
    aug_df = pd.DataFrame(aug_train)
    X_ = pd.concat([X, aug_df])
    X_ = X_.sample(frac=1).reset_index(drop=True)  # shuffle
    X_ = X_["img"].apply(standardize)
    Y_ = X_["font"]
    X_ = X_.drop(columns=["font"])  # drop y again
    return X_, Y_


@log_time
def create_dataset(
    h5_file,
    verbose=False,
    photos=None,
    rotation=False,
    augment=False,
    augment_cycles=3,
    test_size=0.3,
    save=True,
    augment_method=augment_image,
):
    """
    main function - read h5 and return dataset
    """
    logger.info(f"Create dataset started [h5_file={h5_file}]")
    db = h5py.File(h5_file)
    dataset = []
    images = photos or list(db["data"].keys())
    c = 0

    for im in images:
        img = db["data"][im][:]
        metadata = db["data"][im]
        txt = metadata.attrs["txt"]
        fonts = metadata.attrs["font"]
        bbs = metadata.attrs["charBB"]
        chars = mk_charlst(txt)
        assert len(chars) == len(
            fonts
        ), "Some letters do not have their corresponding font"
        for i in range(len(fonts)):
            bb = bbs[:, :, i]
            font = fonts[i]
            char, word = chars[i]

            img_ = crop(img, bb)

            if rotation:
                img_ = rotate(img_, bb)

            dataset.append(
                {
                    "img": img_,
                    "font": font.decode("utf-8"),
                    "char": char,
                    "word": word,
                    "img_name": im,
                }
            )
            if verbose:
                c += 1
                logger.info(f"Finished image {c}/{len(images)} [image={im}]")

    df = pd.DataFrame(dataset)
    Y = df["font"]
    X = df.drop(columns=["font"])
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, random_state=0, test_size=test_size
    )
    if augment:
        # augment train data
        x_train, y_train = _augment_dataset(
            x_train,
            y_train,
            augment_method=augment_method,
            augment_cycles=augment_cycles,
            verbose=verbose,
        )
        x_test, y_test = _augment_dataset(
            x_test,
            y_test,
            augment_method=augment_method,
            augment_cycles=augment_cycles,
            verbose=verbose,
        )

    if save:
        import time

        t = str(int(time.time()))[-3:]  # for unique name
        l_ = len(df)
        aug_cycles = 0 if not augment else augment_cycles

        train_data = x_train
        train_data["font"] = y_train

        test_data = x_test
        x_test["font"] = y_test

        train_data.to_hdf(
            f"db/prep_{l_}r_{aug_cycles}a_train_{augment_method.__name__}_{t}.h5",
            key="db",
        )
        test_data.to_hdf(
            f"db/prep_{l_}r_{aug_cycles}a_test_{augment_method.__name__}_{t}.h5",
            key="db",
        )
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    ds = create_dataset(
        "SynthText.h5",
        verbose=1,
        rotation=True,
        augment=True,
        augment_cycles=1,
        save=True,
        augment_method=augment_image,
    )
