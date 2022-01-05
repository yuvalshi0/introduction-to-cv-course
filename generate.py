import os

import pandas as pd

from config import config

CODA_DLL_PATH = config["preprocessing"]["coda_dll_path"]
os.add_dll_directory(
    CODA_DLL_PATH
)  # https://github.com/tensorflow/tensorflow/issues/48868#issuecomment-841396124


import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

from augment import augment_image
from plogging import log_time, logger
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


@log_time
def generate_images(num_images=20, verbose=False, augment=False, augment_cycles=20):
    dataset = []
    total_images = num_images * len(CHARS) * len(FONTS)
    c = 0
    for _ in range(num_images):
        for char in CHARS:
            for font in FONTS:
                base_img_name = np.random.choice(IMAGES)
                img_ = generate_image(char=char, font=font, base_img_name=base_img_name)
                simg_ = standardize(np.asarray(img_))
                dataset.append(
                    {
                        "img": simg_,
                        "font": font,
                        "char": char,
                        "word": GENERATE_TOKEN,
                        "img_name": GENERATE_TOKEN,
                    }
                )
                if augment:
                    for _ in range(augment_cycles):
                        tmpimg_ = augment_image(img_)
                        simg_ = standardize(tmpimg_)
                        dataset.append(
                            {
                                "img": simg_,
                                "font": font,
                                "char": char,
                                "word": GENERATE_TOKEN,
                                "img_name": GENERATE_TOKEN,
                            }
                        )
                if verbose:
                    c += 1
                    logger.info(f"Finished generating photo {c}/{total_images}")
    df = pd.DataFrame(dataset)
    return df


def generate_image(char, font, base_img_name):
    """
    generate an image
    """
    img = cv2.imread(f"{IMAGE_FOLDER}\\{base_img_name}")

    font_size = int(IMG_SIZE * 1.5)
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
    color = (
        np.random.randint(0, 255),
        np.random.randint(0, 255),
        np.random.randint(0, 255),
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
    import time

    ct = int(time.time())
    for i in range(10):
        acycles = 30
        images = generate_images(
            num_images=50, verbose=1, augment=True, augment_cycles=acycles
        )
        images.to_hdf(f"db/generated_{acycles}_{ct}_{i}.h5", key="db")
