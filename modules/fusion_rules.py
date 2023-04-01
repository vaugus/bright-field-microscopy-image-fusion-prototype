#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module with fusion rule implementations.

This module contains an abstract class to organize any
fusion rule implementation. It also contains a class
that implements the Energy of Laplacian of Gaussian
fusion approach.
"""

import abc
import os

import numpy as np
from scipy import ndimage


class AbstractFocusMeasure(metaclass=abc.ABCMeta):
    """Abstract class for fusion rule implementations."""

    def __init__(self):
        """Constructor."""
        super(AbstractFocusMeasure, self).__init__()

    @abc.abstractmethod
    def execute(self, images, grayscale_images, sigma):
        """Executes the fusion rule."""
        pass


class LaplacianOfGaussianEnergy(AbstractFocusMeasure):
    """Class that implements the Laplacian of Gaussian Energy fusion rule."""

    def __init__(self):
        """Constructor."""
        super().__init__()

    def laplacian(self, image):
        return ndimage.gaussian_laplace(image, sigma=0.7)

    def execute(self, dataset, grayscale_images):
        """Executes the Energy of Laplacian of Gaussian fusion rule.

        :param kwargs['dataset']: List of dataset images.
        :type kwargs['dataset']: list of PIL.Image

        :param kwargs['gray_dataset']: List of grayscale numpy ndarray
        representations of dataset images.
        :type kwargs['gray_dataset']: list of numpy.ndarray

        :param kwargs['sigma']: Standard deviation of the Gaussian function.
        :type kwargs['sigma']: float

        :returns: The fused image.
        :rtype: numpy.ndarray
        """
        # compute the edges of the images with the laplacian filter
        edges = np.array([self.laplacian(img) for img in grayscale_images])

        # compute 3D indices of the highest energies of laplacian,
        # pixel-wise

        energies = np.argmax(edges, axis=0)

        # Synthesize an image by sampling from each layer
        # according to it's highest energy

        height, width = grayscale_images[0].shape

        result = np.zeros((height, width, 3), dtype=np.float32)

        stack = np.stack(dataset, axis=0)
        result = np.zeros((height, width, 3), dtype=np.float32)
        for row in range(height):
            for col in range(width):
                idx = energies[row, col]
                result[row, col, :] = stack[idx, row, col, :]

        return result
