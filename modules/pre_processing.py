#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pathlib

import numpy as np
from PIL import Image


class PreProcessing(object):
    def sort_dataset_names(self, files, extension):
        files = [item.replace(extension, '') for item in files]
        files.sort()

        return [item + extension for item in files]

    def open_dataset(self, path):
        filenames = os.listdir(path)
        extension = pathlib.Path(filenames[0]).suffix

        files = self.sort_dataset_names(filenames, extension)
        return [Image.open(path + image) for image in files]

    def apply_luminance(self, image):
        tmp = self.image_to_ndarray(image)
        gray = tmp[:, :, 0] * .299 + tmp[:, :, 1] * .587 + tmp[:, :, 2] * .114
        return gray

    def convert_images_to_grayscale(self, dataset):
        return [
            self.image_to_ndarray(
                    self.ndarray_to_image(
                        self.apply_luminance(image)))
            for image in dataset
        ]

    def normalize_image(self, image):
        min_, max_ = image.min(), image.max()
        return (image - min_) / (max_ - min_)

    def image_to_ndarray(self, image):
        return np.array(image, dtype=np.float64) / 255.

    def ndarray_to_image(self, array, normalize=True):
        if normalize:
            array = self.normalize_image(array)

        return Image.fromarray((array * 255.).astype(np.uint8))
