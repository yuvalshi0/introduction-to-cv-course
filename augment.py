"""
This file contains the augmations method I described at section V at the report to augment methods
Was not used for the final model
"""
import itertools

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def noisy(noise_typ, image):
    """
    Inspired from https://stackoverflow.com/a/30609854
    """
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[tuple(coords)] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy
    raise ValueError(f"Unknown noise_type={noise_typ}")


def blur(img):
    return cv2.GaussianBlur(img, (3, 3), 0).astype(np.uint8)


def laplacian(img):
    return cv2.Laplacian(np.uint8(img), cv2.CV_64F).astype(np.uint8)


def rotate(img):
    shear = 180 if np.random.randint(0, 11) == 10 else 0  # 10%
    angle = np.random.randint(-45, 45)
    return tf.keras.preprocessing.image.apply_affine_transform(
        img, theta=angle, shear=shear
    )


def random_brightness(img):
    return tf.keras.preprocessing.image.random_brightness(img, (0.5, 1.5)).astype(
        np.uint8
    )


def s_and_p(image):
    """
    Taken partly from https://stackoverflow.com/a/30609854
    adds 50% random blur and s&p to the image
    DID WORSE ON TRAIN
    """

    s_vs_p = 0.5
    amount = 0.004
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[tuple(coords)] = 1

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[tuple(coords)] = 0
    return np.uint8(out)


def noise_image(img):
    # Adding Noise to image
    img_array = np.asarray(img)
    mean = 0.0  # some constant
    std = 5  # some constant (standard deviation)
    noisy_img = img_array + np.random.normal(mean, std, img_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255)
    return np.uint8(noisy_img_clipped)


def zoom(img):
    return tf.keras.preprocessing.image.random_zoom(img, (0.8, 1.2)).astype(np.uint8)


def shift(img):
    return tf.keras.preprocessing.image.random_shift(img, 0.1, 0.1).astype(np.uint8)


def augment_image(record, angle_range=45):
    """
    EXPERIMENTAL augment to add blur also - did worse on tests
    """
    img = record["img"]
    img_ = tf.keras.preprocessing.image.random_brightness(img, (0.7, 1.2)).astype(
        np.uint8
    )
    img_ = tf.keras.preprocessing.image.random_zoom(img_, (0.8, 1)).astype(np.uint8)
    angle = np.random.randint(-angle_range, angle_range)
    shear = 180 if np.random.randint(0, 10) == 11 else 0  # 10%
    img_ = tf.keras.preprocessing.image.apply_affine_transform(
        img_, theta=angle, shear=shear
    )
    img_ = tf.keras.preprocessing.image.random_shift(img_, 0.05, 0.05).astype(np.uint8)
    record_ = dict(record)
    record["img"] = img_
    return [record_]


def augment_image_v4(record):
    augment_methods = [blur, rotate, shift]
    dataset = []
    for l in range(1, len(augment_methods) + 1):
        combs = itertools.combinations(augment_methods, l)
        for comb in combs:  # we don't want to just do s&p
            tmpimg = record["img"]
            aug_comb = list(comb)
            for aug_method in aug_comb:
                tmpimg = aug_method(tmpimg)
            dataset.append(
                {
                    "img": tmpimg,
                    "font": record["font"],
                    "char": record["char"],
                    "word": record["word"],
                    "img_name": record["img_name"],
                }
            )
    return dataset


def canny(img):
    img_ = cv2.GaussianBlur(img, (5, 5), 0).astype(np.uint8)
    img_ = cv2.Canny(img_, 50, 200, apertureSize=5)
    return img_


def sobel(img):
    h = cv2.Sobel(img, 0, 1, 0, cv2.CV_64F)
    v = cv2.Sobel(img, 0, 0, 1, cv2.CV_64F)
    return cv2.bitwise_or(h, v).astype(np.uint8)


def none(img):
    return img


def augment_image_v5(record):
    dataset = []
    tmpimg = record["img"]
    augment_methods = [laplacian, blur, noise_image]
    for l in range(1, len(augment_methods) + 1):
        aug_combs = itertools.combinations(augment_methods, l)
        for aug_comb in aug_combs:
            tmpimg_ = tmpimg
            for method in aug_comb:
                tmpimg_ = method(tmpimg_)
            tmpimg_ = cv2.cvtColor(tmpimg, cv2.COLOR_BGR2GRAY)
            dataset.append(
                {
                    "img": tmpimg_,
                    "font": record["font"],
                    "char": record["char"],
                    "word": record["word"],
                    "img_name": record["img_name"],
                }
            )
    return dataset
