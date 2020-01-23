#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import *
from pytictoc import TicToc
from PIL import Image, ImageFilter, ImageOps
import skimage
import numpy as np

from sklearn.preprocessing import MinMaxScaler


def open_image(path):
    """Method and function names are lower_case_with_underscores.

    Always use self as first arg.
    """
    img = Image.open(path)
    return img

def sort_dataset_names(files):
    files = [item.replace('.jpg', '') for item in files]
    files.sort()
    files = [item + '.jpg' for item in files]
    
    return files

def open_dataset(path):
    """Method and function names are lower_case_with_underscores.

    Always use self as first arg.
    """
    files = sort_dataset_names(os.listdir(path))
    return [open_image(path + img) for img in files]


def normalize_image(arr):
    """Method and function names are lower_case_with_underscores.

    Always use self as first arg.
    """
    min_, max_ = arr.min(), arr.max()
    return (arr - min_) / (max_ - min_)


def to_array(img):
    """Method and function names are lower_case_with_underscores.

    Always use self as first arg.
    """
    return np.array(img, dtype=np.float32) / 255.


def to_image(arr, normalize):
    """Method and function names are lower_case_with_underscores.

    Always use self as first arg.
    """
    if normalize:
        arr = normalize_image(arr) 
    
    return Image.fromarray((arr * 255.).astype(np.uint8))


def grayscale_averaging(img):
    """Method and function names are lower_case_with_underscores.

    Always use self as first arg.
    """
    gray = to_array(img).mean(axis=2)
    return to_image(gray, False)


def grayscale_gleam(img):
    """Method and function names are lower_case_with_underscores.

    Always use self as first arg.
    """
    gray = np.power(to_array(img), 1/2.2).mean(axis=2)
    return to_image(gray, False)


def main():
    pass

if __name__ == "__main__":
	main()
