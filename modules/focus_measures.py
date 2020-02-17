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

import abc
import os

import numpy as np
from scipy import ndimage

class AbstractFocusMeasure(metaclass=abc.ABCMeta):
    """Class with pre-processing operations for fusion rules."""

    def __init__(self):
        """Constructor."""
        super(AbstractFocusMeasure, self).__init__()

    @abc.abstractmethod
    def execute(self, **kwargs):
        pass


class EnergyOfLaplacian(AbstractFocusMeasure):
    """Class with pre-processing operations for fusion rules."""

    def __init__(self):
        """Constructor."""
        super().__init__()


    def laplacian_of_gaussian_filter(self, arr, sigma=3, square=False):
        """Method and function names are lower_case_with_underscores.

        Always use self as first arg.
        """
        gradients = ndimage.gaussian_laplace(arr, sigma=sigma)
        if square:
            gradients *= gradients

        return gradients


    def execute(self, **kwargs):
        """Method and function names are lower_case_with_underscores.

        Always use self as first arg.
        """
        dataset = kwargs['dataset']
        gray_dataset = kwargs['gray_dataset']
        sigma = kwargs['sigma']

        # compute the edges of the images with the laplacian filter
        edges = np.stack([self.laplacian_of_gaussian_filter(arr, sigma, square=False)
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
