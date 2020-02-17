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

import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk


class Evaluation(object):
    """Class with pre-processing operations for fusion rules."""

    def __init__(self):
        super().__init__()

    def spatial_frequency(self, img):
        x, y, z = img.shape
        RF = 0.0
        CF = 0.0
        for i in range(x):
            for j in range(1, y):
                for k in range(z):
                    RF += np.square(img[i, j, k] - img[i, j-1, k])

        for i in range(1, x):
            for j in range(y):
                for k in range(z):
                    CF += np.square(img[i, j, k] - img[i-1, j, k])

        CF = np.sqrt((1 / img.size) * CF)
        RF = np.sqrt((1 / img.size) * RF)

        return np.sqrt(np.square(RF) + np.square(CF))

    def STD(self, img):
        n, bins = np.histogram(img)

        mids = 0.5*(bins[1:] + bins[:-1])
        mean = np.average(mids, weights=n)
        var = np.average((mids - mean)**2, weights=n)

        return np.sqrt(var)

    def entropy(self, RGB):
        r = np.array(RGB).astype(np.uint8)[:,:, 0]
        g = np.array(RGB).astype(np.uint8)[:,:, 1]
        b = np.array(RGB).astype(np.uint8)[:,:, 2]

        H = entropy(r, disk(3)) + entropy(g, disk(3)) + entropy(b, disk(3))
        return np.mean(H)
