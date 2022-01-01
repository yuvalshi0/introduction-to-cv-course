import os

import pandas as pd

from config import config

CODA_DLL_PATH = config["preprocessing"]["coda_dll_path"]
os.add_dll_directory(
    CODA_DLL_PATH
)  # https://github.com/tensorflow/tensorflow/issues/48868#issuecomment-841396124


import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

from plogging import log_time
from preprocessing import standardize

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
IMAGE_FOLDER = config["generate"]["images_folder"]
IMAGES = [img for img in os.listdir(IMAGE_FOLDER) if img.endswith(".png")]

FONTS_FOLDER = config["generate"]["fonts_folder"]
FONTS = config.get_classes()
IMG_SIZE = int(config["main"]["img_size"])
GENERATE_TOKEN = "__generated__"

CHARS = [
    "!",
    '"',
    "#",
    "$",
    "%",
    "&",
    "'",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    ";",
    "<",
    ">",
    "?",
    "@",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "[",
    "\\",
    "]",
    "^",
    "_",
    "`",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "|",
]


def _is_color_dark(color):
    r, g, b = color
    hsp = np.sqrt(0.299 * (r * r) + 0.587 * (g * g) + 0.114 * (b * b))
    return hsp <= 127.5


def noisy(noise_typ, image):
    """
    Taken from https://stackoverflow.com/a/30609854
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


def augment_image(img, noise_type="s&p", angle_range=90):
    img_ = tf.keras.preprocessing.image.random_brightness(img, (0.7, 1.3)).astype(
        np.uint8
    )
    img_ = tf.keras.preprocessing.image.random_zoom(img_, (1, 1)).astype(np.uint8)
    angle = np.random.randint(-angle_range, angle_range)
    # theta = 0
    shear = 180 if np.random.randint(1, 10) == 10 else 0  # 10% to be 18
    # shear = 0
    img_ = tf.keras.preprocessing.image.apply_affine_transform(
        img_, theta=angle, shear=shear
    )
    img_ = noisy(noise_type, img_)
    return img_


@log_time
def generate_images(num_images=20):
    dataset = []
    for _ in range(num_images):
        for char in CHARS:
            for font in FONTS:
                base_img_name = np.random.choice(IMAGES)
                img = generate_image(char=char, font=font, base_img_name=base_img_name)
                img_ = augment_image(img)
                img_ = standardize(img_)
                dataset.append(
                    {
                        "img": img_,
                        "font": font,
                        "char": char,
                        "word": GENERATE_TOKEN,
                        "img_name": GENERATE_TOKEN,
                    }
                )
    df = pd.DataFrame(dataset)
    return df


def generate_image(char, font, base_img_name):
    """
    generate an image
    """
    img = cv2.imread(f"{IMAGE_FOLDER}\\{base_img_name}")

    font_size = 50
    font_ = ImageFont.truetype(f"{FONTS_FOLDER}\\{font}.ttf", font_size)
    # random start points
    start_height = np.random.randint(0, img.shape[0] - IMG_SIZE)
    start_width = np.random.randint(0, img.shape[1] - IMG_SIZE)
    img_ = Image.fromarray(
        img[
            start_height : start_height + IMG_SIZE,
            start_width : start_width + IMG_SIZE,
        ]
    )
    d = ImageDraw.Draw(img_)
    w, h = d.textsize(char, font=font_)
    while w > IMG_SIZE or h > IMG_SIZE:
        font_size -= 2
        font_ = ImageFont.truetype(f"{FONTS_FOLDER}\\{font}.ttf", font_size)
        w, h = d.textsize(char, font=font_)
    # image random color
    mean_c = np.mean(img_, axis=(0, 1))
    color = (
        (
            np.random.randint(0, 125),
            np.random.randint(0, 125),
            np.random.randint(0, 125),
        )
        if not _is_color_dark(mean_c)
        else (
            np.random.randint(125, 255),
            np.random.randint(125, 255),
            np.random.randint(125, 255),
        )
    )
    d.text(
        (
            (IMG_SIZE - w + w * np.random.uniform(-0.1, 0.1)) / 2,
            (IMG_SIZE - h + h * np.random.uniform(-0.1, 0.1)) / 2,
        ),
        text=char,
        fill=color,
        font=font_,
    )
    return img_


if __name__ == "__main__":
    generate_images(num_images=10)
