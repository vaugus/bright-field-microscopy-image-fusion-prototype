#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module with a wrapper for fusion rules.

This module contains a facade class to wrap the proposed fusion
rule implementation and eventually new implementions.
"""

import os

import numpy as np
from PIL import Image

from .pre_processing import PreProcessing
from .fusion_rules import LaplacianOfGaussianEnergy
from .evaluation import Evaluation


class Fusion(object):
    """Facade class to wrap the execution of fusion rules."""

    def __init__(self):
        """Constructor.

        This method instantiates all the objects to perform the
        fusion procedure.

        Attributes:
            pre_processing      PreProcessing object.
            LoG_energy          LaplacianOfGaussianEnergy object.
            evaluation          Evaluation object.
        """
        super().__init__()
        self.pre_processing = PreProcessing()
        self.LoG_energy = LaplacianOfGaussianEnergy()
        self.evaluation = Evaluation()

    def run(self, path):
        """Performs the image fusion procedure.

        :param path: Location of the dataset in the filesystem.
        :type path: str
        """
        size = None

        # open the dataset images
        dataset = self.pre_processing.open_dataset(path, size)

        # convert images to grayscale
        gray_dataset = self.pre_processing.grayscale_dataset(
            dataset, 'luminance')

        sigma = 0.7

        # perform the image fusion
        result = self.LoG_energy.execute(
            dataset=dataset, gray_dataset=gray_dataset, sigma=sigma)

        A = self.pre_processing.ndarray_to_image(result)
        result = self.pre_processing.normalize_image(result)

        # Evaluate
        print('SF: {0}'.format(self.evaluation.spatial_frequency(result)))
        print('STD: {0}'.format(self.evaluation.STD(result)))
        print('Entropy: {0}'.format(self.evaluation.entropy(A)))
