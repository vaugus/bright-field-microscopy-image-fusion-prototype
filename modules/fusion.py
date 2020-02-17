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
from PIL import Image

from modules.pre_processing import PreProcessing
from modules.focus_measures import EnergyOfLaplacian
from modules.evaluation import Evaluation

class Fusion(object):
    """Class with pre-processing operations for fusion rules."""


    def __init__(self):
        super().__init__()
        self.pre_processing = PreProcessing()
        self.energy_of_laplacian = EnergyOfLaplacian()
        self.evaluation = Evaluation()

    def run(self, path):
        """Method and function names are lower_case_with_underscores.

        Always use self as first arg.
        """
        # open the dataset images
        size = None

        dataset = self.pre_processing.open_dataset(path, size)

        # # convert images to grayscale
        gray_dataset = self.pre_processing.grayscale_dataset(dataset, 'luminance')

        sigma = 0.7

        result = self.energy_of_laplacian.execute(
            dataset=dataset, gray_dataset=gray_dataset, sigma=sigma)

        # p = '/home/victor/Desktop/LOG.tif'
        # p = '/home/victor/Desktop/PCA.tif'
        # p = '/home/victor/Desktop/GF.tif'
        # p = '/home/victor/Desktop/MSGW.tif'
        # p = '/home/victor/Desktop/MSVD.tif'
        # A = self.pre_processing.open_image(p, None)

        # result = self.pre_processing.image_to_ndarray(A)
        A = self.pre_processing.ndarray_to_image(result)
        result = self.pre_processing.normalize_image(result)
        
        print('SF: {0}'.format(self.evaluation.spatial_frequency(result)))
        print('STD: {0}'.format(self.evaluation.STD(result)))
        print('Entropy: {0}'.format(self.evaluation.entropy(A)))
