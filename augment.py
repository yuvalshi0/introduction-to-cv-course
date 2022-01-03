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


def _odd(n):
    return n if n % 2 != 0 else n + 1


def noisy_v2(image):
    """
    Taken partly from https://stackoverflow.com/a/30609854
    adds 50% random blur and s&p to the image
    DID WORSE ON TRAIN
    """
    if np.random.randint(0, 3) == 1:
        kernel_size = _odd(np.random.randint(1, 6))
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

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


def augment_image(img, noise_type="s&p", angle_range=90):
    """
    Adds random zoom, 10% random shear, random brightness, random shift and s&p noise
    """
    img_ = tf.keras.preprocessing.image.random_brightness(img, (0.5, 1.5)).astype(
        np.uint8
    )
    img_ = tf.keras.preprocessing.image.random_zoom(img_, (1, 1)).astype(np.uint8)
    angle = np.random.randint(-angle_range, angle_range)
    shear = 180 if np.random.randint(0, 11) == 10 else 0  # 10%
    img_ = tf.keras.preprocessing.image.apply_affine_transform(
        img_, theta=angle, shear=shear
    )
    img_ = tf.keras.preprocessing.image.random_shift(img_, 0.05, 0.05).astype(np.uint8)
    img_ = noisy(noise_type, img_)
    return img_


def augment_image_v2(img, angle_range=90):
    """
    EXPERIMENTAL augment to add blur also - did worse on tests
    """
    img_ = tf.keras.preprocessing.image.random_brightness(img, (0.5, 1.5)).astype(
        np.uint8
    )
    img_ = tf.keras.preprocessing.image.random_zoom(img_, (1, 1)).astype(np.uint8)
    angle = np.random.randint(-angle_range, angle_range)
    shear = 180 if np.random.randint(0, 10) == 11 else 0  # 10%
    img_ = tf.keras.preprocessing.image.apply_affine_transform(
        img_, theta=angle, shear=shear
    )
    img_ = tf.keras.preprocessing.image.random_shift(img_, 0.05, 0.05).astype(np.uint8)
    img_ = noisy_v2(img_)
    return img_


def BAD_augment_image(img, noise_type="s&p", angle_range=45):
    """
    BAD AUGMENTATION, see section 4 in the report
    """
    img_ = tf.keras.preprocessing.image.random_brightness(img, (0.7, 1.3)).astype(
        np.uint8
    )
    img_ = tf.keras.preprocessing.image.random_zoom(img_, (0, 1)).astype(np.uint8)
    img_ = tf.keras.preprocessing.image.random_rotation(img_, angle_range)
    img_ = tf.keras.preprocessing.image.random_shear(img_, 180)
    img_ = tf.keras.preprocessing.image.random_shift(img_, 0.1, 0.1)
    img_ = noisy(noise_type, img_)
    return img_
