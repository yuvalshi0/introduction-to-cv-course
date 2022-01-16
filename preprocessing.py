import os

from config import config

CODA_DLL_PATH = config["preprocessing"]["coda_dll_path"]
os.add_dll_directory(
    CODA_DLL_PATH
)  # https://github.com/tensorflow/tensorflow/issues/48868#issuecomment-841396124


import pathlib

import cv2
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from plogging import log_time, logger

IMG_SIZE = int(config["main"]["img_size"])
CLASSES = config.get_classes()
CACHE_PATH = config["main"]["cache_path"]


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
    resize to img size
    """
    # if len(img.shape) > 2:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))


def _decode(s):
    return s.decode("utf-8") if isinstance(s, bytes) else s


def mk_charlst(lst):
    """
    helper function, returns a list of tuples, where the first element is the char and the second element is the word
    """
    clst = []
    for word in lst:
        word_ = _decode(word)
        clst += [(c, word_) for c in word_]
    return clst


def _augment_dataset(X, Y, augment_method, verbose=False):
    X["font"] = np.array(Y)  # temporary unite the datasets
    records = X.to_dict("records")
    aug_train = []
    c = 0
    for record in records:
        aug_train += augment_method(record)
        if verbose:
            c += 1
            img_name = record["img_name"]
            logger.info(
                f"Finished augmenting image {c}/{len(records)} [image={img_name}]"
            )
    aug_df = pd.DataFrame(aug_train)
    X_ = pd.concat([X, aug_df])
    X_ = X_.sample(frac=1).reset_index(drop=True)  # shuffle
    Y_ = X_["font"]
    X_ = X_.drop(columns=["font"])  # drop y again
    return X_, Y_


def create_base_dataset(db, rotation, photos, verbose):
    dataset = []
    c = 0
    images = photos or list(db["data"].keys())
    for im in images:
        try:
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
                img_ = standardize(img_)
                dataset.append(
                    {
                        "img": img_,
                        "font": _decode(font),
                        "char": char,
                        "word": word,
                        "img_name": im,
                    }
                )
        except Exception as e:
            logger.info(f"Falied to add the image: {im}, failure reason: {e}")
        if verbose:
            c += 1
            logger.info(f"Finished image {c}/{len(images)} [image={im}]")
    return pd.DataFrame(dataset)


@log_time
def create_dataset(
    h5_file,
    verbose=False,
    photos=None,
    rotation=True,
    test_size=0.3,
    save=False,
    no_split=False,
    cache=False,
):
    """
    Main method - preprocess & create the dataset

    Args:
        h5_file (str): path to the h5 file dataset
        verbose (bool, optional): print out the method process. Defaults to False.
        photos (List[str], optional): preprocess only if fixed list of photos, useful for debugging, if None - will preprocess all the pictures. Defaults to None.
        rotation (bool, optional): Rotate the images. Defaults to True.
        test_size (float, optional): Test size split. Defaults to 0.3.
        save (bool, optional): Save the preprocessed data to the db Folder. Defaults to False.
        no_split (bool, optional): Don't split the dataset into train and test data, just return X and y. Defaults to False.
        cache (bool, optional): Cache the dataset, useful when you don't want to preprocess the same dataset again and again. Defaults to False.

    Returns:
        (train_x, test_x),(test_x, test_y) OR (X,Y) depends if no_split was given True or False
    """
    logger.info(f"Create dataset started [h5_file={h5_file}]")
    db = h5py.File(h5_file)
    cache_path = CACHE_PATH.format(IMG_SIZE)
    if cache and os.path.exists(cache_path):
        df = pd.read_hdf(cache_path, key="db")
        logger.info(
            f"Collected db from catche [cache_path={CACHE_PATH.format(IMG_SIZE)}]"
        )
    else:
        df = create_base_dataset(db, rotation, photos, verbose)
        if cache:
            pathlib.Path("db").mkdir(exist_ok=True)
            df.to_hdf(cache_path, key="db")

    Y = df["font"]
    X = df.drop(columns=["font"])
    X["img"]
    if no_split:
        if save:
            import time

            t = str(int(time.time()))[-3:]  # for unique name
            l_ = len(df)
            df.to_hdf(
                f"db/prep_{l_}r_unsplit_{t}.h5",
                key="db",
            )
        return X, Y
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, random_state=0, test_size=test_size
    )
    if save:
        import time

        t = str(int(time.time()))[-3:]  # for unique name
        l_ = len(df)

        train_data = x_train
        train_data["font"] = y_train

        test_data = x_test
        x_test["font"] = y_test

        train_data.to_hdf(
            f"db/prep_{l_}r_train_{t}.h5",
            key="db",
        )
        test_data.to_hdf(
            f"db/prep_{l_}r_test_{t}.h5",
            key="db",
        )
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    H5_FILE = config["main"]["h5_file"]
    ds = create_dataset(H5_FILE, verbose=1, save=True, use_cache=True)
