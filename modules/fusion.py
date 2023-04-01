#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module with a wrapper for fusion rules.

This module contains a facade class to wrap the proposed fusion
rule implementation.
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
        dataset = self.pre_processing.open_dataset(path)

        gray_dataset = self.pre_processing.convert_images_to_grayscale(dataset)

        result = self.LoG_energy.execute(dataset, gray_dataset)

        A = self.pre_processing.ndarray_to_image(result)
        result = self.pre_processing.normalize_image(result)

        sf = self.evaluation.spatial_frequency(result)
        std = self.evaluation.STD(result)
        entropy = self.evaluation.entropy(A)

        # Evaluate
        print('SF: {0}'.format(sf))
        print('STD: {0}'.format(std))
        print('Entropy: {0}'.format(entropy))
