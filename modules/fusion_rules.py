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
    def execute(self, **kwargs):
        """Executes the fusion rule."""
        pass


class LaplacianOfGaussianEnergy(AbstractFocusMeasure):
    """Class that implements the Laplacian of Gaussian Energy fusion rule."""

    def __init__(self):
        """Constructor."""
        super().__init__()

    def laplacian_of_gaussian_filter(self, img, sigma, square=False):
        """Performs the Laplacian of Gaussian filtering process.

        :param img: The image to be filtered.
        :type img: numpy.ndarray

        :param sigma: The standard deviation of the Gaussian function.
        :type sigma: float

        :param square: A flag for squaring the gradients or not.
        :type square: boolean

        :returns: The second-order derivatives of the image.
        :rtype: numpy.ndarray
        """
        gradients = ndimage.gaussian_laplace(img, sigma=sigma)
        if square:
            gradients *= gradients

        return gradients

    def execute(self, **kwargs):
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
        dataset = kwargs['dataset']
        gray_dataset = kwargs['gray_dataset']
        sigma = kwargs['sigma']

        # compute the edges of the images with the laplacian filter
        edges = np.stack([self.laplacian_of_gaussian_filter(arr, sigma)
                          for arr in gray_dataset], axis=2)

        # compute 3D indices of the highest energies of laplacian,
        # pixel-wise
        energies = np.argmax(edges, axis=2)
        # Synthesize an image by sampling from each layer
        # according to it's highest energy

        result = None
        height, width = edges.shape[0:2]

        stack = np.stack(dataset, axis=0)
        result = np.zeros((height, width, 3), dtype=np.float32)
        for row in range(height):
            for col in range(width):
                idx = energies[row, col]
                result[row, col, :] = stack[idx, row, col, :]

        return result
