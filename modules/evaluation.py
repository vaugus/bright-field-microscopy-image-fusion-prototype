#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk as disk_footprint

from itertools import product


class Evaluation(object):
    def spatial_frequency(self, img):
        x, y, z = img.shape
        RF = 0.0
        CF = 0.0

        for i, j, k in product(range(x), range(y), range(z)):
            if j > 1:
                RF += np.square(img[i, j, k] - img[i, j-1, k])

            if i > 1:
                CF += np.square(img[i, j, k] - img[i-1, j, k])

        CF = np.sqrt((1 / img.size) * CF)
        RF = np.sqrt((1 / img.size) * RF)

        return np.sqrt(np.square(RF) + np.square(CF))

    def STD(self, img):
        n, bins = np.histogram(img)

        mids = 0.5 * (bins[1:] + bins[:-1])
        mean = np.average(mids, weights=n)
        var = np.average(np.square(mids - mean), weights=n)

        return np.sqrt(var)

    def entropy(self, rgb_image):
        r = np.array(rgb_image, dtype=np.uint8)[:, :, 0]
        g = np.array(rgb_image, dtype=np.uint8)[:, :, 1]
        b = np.array(rgb_image, dtype=np.uint8)[:, :, 2]

        disk = disk_footprint(3)
        return np.mean(entropy(r, disk) + entropy(g, disk) + entropy(b, disk))
