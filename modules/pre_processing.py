#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module's docstring summary line.

This is a multi-line docstring. Paragraphs are separated with blank lines.
Lines conform to 79-column limit.

Module and packages names should be short, lower_case_with_underscores.
Notice that this in not PEP8-cheatsheet.py

Seriously, use flake8. Atom.io with https://atom.io/packages/linter-flake8
is awesome!

See http://www.python.org/dev/peps/pep-0008/ for more PEP-8 details
"""

import os

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import skimage


class PreProcessing(object):
    """Class with pre-processing operations for fusion rules."""

    def __init__(self):
        super().__init__()

    def open_image(self, path, size):
        """Method and function names are lower_case_with_underscores.

        Always use self as first arg.
        """
        img = Image.open(path)

        if size:
            img.thumbnail(size, Image.ANTIALIAS)

        return img


    def sort_dataset_names(self, files, extension):
        files = [item.replace(extension, '') for item in files]
        files.sort()
        files = [item + extension for item in files]
        
        return files

    def open_dataset(self, path, size):
        """Method and function names are lower_case_with_underscores.

        Always use self as first arg.
        """
        filenames = os.listdir(path)
        sample_file = filenames[0]
        extension = sample_file[sample_file.find('.') : len(sample_file)]

        files = self.sort_dataset_names(filenames, extension)
        return [self.open_image(path + img, size) for img in files]


    def normalize_image(self, arr):
        """Method and function names are lower_case_with_underscores.

        Always use self as first arg.
        """
        min_, max_ = arr.min(), arr.max()
        return (arr - min_) / (max_ - min_)


    def image_to_ndarray(self, img):
        """Method and function names are lower_case_with_underscores.

        Always use self as first arg.
        """
        return np.array(img, dtype=np.float32) / 255.


    def ndarray_to_image(self, arr, normalize=True):
        """Method and function names are lower_case_with_underscores.

        Always use self as first arg.
        """
        if normalize:
            arr = self.normalize_image(arr) 
        
        return Image.fromarray((arr * 255.).astype(np.uint8))


    def grayscale_averaging(self, img):
        """Method and function names are lower_case_with_underscores.

        Always use self as first arg.
        """
        gray = self.image_to_ndarray(img).mean(axis=2)
        return self.ndarray_to_image(gray)


    def grayscale_luminance(self, img):
        tmp = self.image_to_ndarray(img)
        gray = tmp[:, :, 0] * 0.299 + tmp[:, :, 1] * 0.587 + tmp[:, :, 2] * 0.114
        return self.ndarray_to_image(gray)


    def grayscale_gleam(self, img):
        """Method and function names are lower_case_with_underscores.

        Always use self as first arg.
        """
        gray = np.power(self.image_to_ndarray(img), 1/2.2).mean(axis=2)
        return self.ndarray_to_image(gray)
