#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module with objective evaluation metrics.

This module contains objective evaluation metrics for
image fusion techniques.
"""

import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk


class Evaluation(object):
	"""Class with evaluation tools for fusion rules."""

	def __init__(self):
		"""Constructor."""
		super().__init__()

	def spatial_frequency(self, img):
		"""Computes the Spatial Frequency index for a fused image.

		This method implements the Spatial Frequency index as proposed
		in

		Naidu, V.P.S. and Raol, J.R., 2008.
		"Pixel-level image fusion using wavelets and principal component analysis."
		Defence Science Journal, 58(3), p.338.

		:param img: The fused image to be evaluated.
		:type img: numpy.ndarray

		:returns: The Spatial Frequency index.
		:rtype: float
		"""
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
		"""Computes the standard deviation of the histogram of an image.

		This method implements the standard deviation index as proposed
		in

		Naidu, V.P.S. and Raol, J.R., 2008.
		"Pixel-level image fusion using wavelets and principal component analysis."
		Defence Science Journal, 58(3), p.338.

		:param img: The fused image to be evaluated.
		:type img: numpy.ndarray

		:returns: The STD index.
		:rtype: float
		"""
		n, bins = np.histogram(img)

		mids = 0.5*(bins[1:] + bins[:-1])
		mean = np.average(mids, weights=n)
		var = np.average((mids - mean)**2, weights=n)

		return np.sqrt(var)

	def entropy(self, RGB):
		"""Computes the Entropy of an image.

		This method implements the Entropy index as proposed
		in

		Naidu, V.P.S. and Raol, J.R., 2008.
		"Pixel-level image fusion using wavelets and principal component analysis."
		Defence Science Journal, 58(3), p.338.

		:param img: The fused image to be evaluated.
		:type img: numpy.ndarray

		:returns: The Entropy index.
		:rtype: float
		"""
		r = np.array(RGB).astype(np.uint8)[:, :, 0]
		g = np.array(RGB).astype(np.uint8)[:, :, 1]
		b = np.array(RGB).astype(np.uint8)[:, :, 2]

		H = entropy(r, disk(3)) + entropy(g, disk(3)) + entropy(b, disk(3))
		return np.mean(H)
