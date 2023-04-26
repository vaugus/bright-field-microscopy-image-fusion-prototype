#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import os

import numpy as np
from scipy import ndimage


class AbstractFocusMeasure(metaclass=abc.ABCMeta):
    def __init__(self):
        super(AbstractFocusMeasure, self).__init__()

    @abc.abstractmethod
    def execute(self, images, grayscale_images, sigma):
        pass


class LaplacianOfGaussianEnergy(AbstractFocusMeasure):

    def __init__(self):
        super().__init__()

    def laplacian(self, image):
        return ndimage.gaussian_laplace(image, sigma=0.7)

    def execute(self, dataset, grayscale_images):
        edges = np.array([self.laplacian(img) for img in grayscale_images])

        energies = np.argmax(edges, axis=0)

        height, width = grayscale_images[0].shape

        result = np.zeros((height, width, 3), dtype=np.float32)

        stack = np.stack(dataset, axis=0)
        result = np.zeros((height, width, 3), dtype=np.float32)
        for row in range(height):
            for col in range(width):
                idx = energies[row, col]
                result[row, col, :] = stack[idx, row, col, :]

        return result
