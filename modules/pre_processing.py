#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module with pre-processing tools for image fusion.

This module contains a class that implements all the pre-processing
methods used in our image fusion approach.
"""

import os

import numpy as np
from PIL import Image


class PreProcessing(object):
    """Class with pre-processing operations for fusion rules."""

    def __init__(self):
        """Constructor."""
        super().__init__()

    def open_image(self, path, size=None):
        """Opens an image file from the filesystem.

        :param path: Location of the image in the filesystem.
        :type path: str

        :param size: Size of the image, if the user wants it to be 
        resized before the computation. Default is None.
        :type size: tuple

        :returns: The RGB image read from the file.
        :rtype: PIL.Image
        """

        img = Image.open(path)

        if size:
            img.thumbnail(size, Image.ANTIALIAS)

        return img

    def sort_dataset_names(self, files, extension):
        """Sorts the list of dataset image names.

        :param files: The list of file names.
        :type files: list of str

        :param extension: The extension of the image files. 
        :type size: str

        :returns: The sorted list of dataset image names.
        :rtype: list of str
        """

        files = [item.replace(extension, '') for item in files]
        files.sort()
        files = [item + extension for item in files]

        return files

    def open_dataset(self, path, size):
        """Opens the image dataset from the given path.

        :param path: Location of the dataset in the filesystem.
        :type path: str

        :param size: Size of the images, if the user wants it to be 
        resized before the computation. Default is None.
        :type size: tuple

        :returns: A list of the read RGB images.
        :rtype: list of PIL.Image
        """

        filenames = os.listdir(path)
        sample_file = filenames[0]
        extension = sample_file[sample_file.find('.'): len(sample_file)]

        files = self.sort_dataset_names(filenames, extension)
        return [self.open_image(path + img, size) for img in files]

        def grayscale_lightness(self, img):
            """Converts a RGB image to grayscale with the lightness method.

            :param img: The image to be converted.
            :type img: PIL.Image

            :returns: The grayscale image.
            :rtype: PIL.Image
            """
            tmp = self.image_to_ndarray(img)
            gray = tmp[:, :, 0] * 0.21 + tmp[:, :, 1] * \
                0.71 + tmp[:, :, 2] * 0.07
            return self.ndarray_to_image(gray)

    def grayscale_averaging(self, img):
        """Converts a RGB image to grayscale with the averaging method.

        :param img: The image to be converted.
        :type img: PIL.Image

        :returns: The grayscale image.
        :rtype: PIL.Image
        """
        gray = self.image_to_ndarray(img).mean(axis=2)
        return self.ndarray_to_image(gray)

    def grayscale_luminance(self, img):
        """Converts a RGB image to grayscale with the luminance method.

        :param img: The image to be converted.
        :type img: PIL.Image

        :returns: The grayscale image.
        :rtype: PIL.Image
        """
        tmp = self.image_to_ndarray(img)
        gray = tmp[:, :, 0] * 0.299 + tmp[:, :, 1] * \
            0.587 + tmp[:, :, 2] * 0.114
        return self.ndarray_to_image(gray)

    def grayscale_gleam(self, img):
        """Converts a RGB image to grayscale with the gleam method.

        :param img: The image to be converted.
        :type img: PIL.Image

        :returns: The grayscale image.
        :rtype: PIL.Image
        """
        gray = np.power(self.image_to_ndarray(img), 1/2.2).mean(axis=2)
        return self.ndarray_to_image(gray)

    def grayscale_dataset(self, dataset, method):
        """Converts the RGB images of a dataset to grayscale.

        :param dataset: List of dataset images.
        :type dataset: list of PIL.Image

        :param method: A string to select the conversion method. 
        :type method: str

        :returns: A list of the converted grayscale images.
        :rtype: list of ndarray
        """
        converter = None
        if method == 'luminance':
            converter = self.grayscale_luminance

        if method == 'gleam':
            converter = self.grayscale_gleam

        if method == 'averaging':
            converter = self.grayscale_averaging

        if method == 'lightness':
            converter = self.grayscale_lightness

        return [self.image_to_ndarray(
            converter(img)) for img in dataset]

    def normalize_image(self, arr):
        """Performs a min-max normalization on the image values.

        :param arr: The array of image pixels.
        :type arr: numpy.ndarray

        :returns: The normalized image as an array.
        :rtype: numpy.ndarray
        """
        min_, max_ = arr.min(), arr.max()
        return (arr - min_) / (max_ - min_)

    def image_to_ndarray(self, img):
        """Converts a PIL.Image object to a numpy.ndarray object.

        :param img: The image to be converted.
        :type img: PIL.Image

        :returns: The numpy.ndarray representation of the image.
        :rtype: numpy.ndarray
        """
        return np.array(img, dtype=np.float32) / 255.

    def ndarray_to_image(self, arr, normalize=True):
        """Converts a numpy.ndarray object to a PIL.Image object.

        :param img: The numpy.ndarray representation of the image 
        to be converted.
        :type img: numpy.ndarray

        :returns: The converted array.
        :rtype: PIL.Image
        """
        if normalize:
            arr = self.normalize_image(arr)

        return Image.fromarray((arr * 255.).astype(np.uint8))
