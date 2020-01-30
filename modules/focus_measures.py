import abc
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, signal


class AbstractFocusMeasure(metaclass=abc.ABCMeta):

    def __init__(self):
        super(AbstractFocusMeasure, self).__init__()

    @abc.abstractmethod
    def execute(self, **kwargs):
        pass


class EnergyOfLaplacian(AbstractFocusMeasure):

    def __init__(self):
        super().__init__()

    def laplacian_filter(self, arr, kernel_size=3, square=True):
        gradients = ndimage.gaussian_laplace(arr, sigma=11)
        # gradients = skimage.filters.laplace(arr, kernel_size)

        if square:
            gradients *= gradients

        return gradients


    def laplacian_of_gaussian_filter(self, arr, sigma=3, square=False):
        gradients = ndimage.gaussian_laplace(arr, sigma=sigma)

        if square:
            gradients *= gradients

        return gradients

    def diagonal_laplacian_filter(self, arr, alpha, square=True):
        lap = np.array([[0, 1, 0],[1, -4, 1], [0, 1, 0]])
        lap = (1.0 - alpha) / (1.0 + alpha) * lap + (alpha) / (1.0 + alpha) * lap

        # print ndimage.convolve(image, stencil, mode='wrap')
        # gradients = skimage.filters.laplace(arr, kernel_size)
        gradients = signal.convolve2d(arr, lap, mode='same')

        if square:
            gradients *= gradients

        return gradients


    def execute(self, **kwargs):
        dataset = kwargs['dataset']
        gray_dataset = kwargs['gray_dataset']

        # compute the edges of the images with the laplacian filter
        edges = np.stack([self.laplacian_of_gaussian_filter(arr, 10000, square=False)
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



# def LAPV(img):
#     """Implements the Variance of Laplacian (LAP4) focus measure
#     operator. Measures the amount of edges present in the image.
#     :param img: the image the measure is applied to
#     :type img: numpy.ndarray
#     :returns: numpy.float32 -- the degree of focus
#     """
#     return numpy.std(skimage.filters.laplace(img)) ** 2


# def LAPM(img):
#     """Implements the Modified Laplacian (LAP2) focus measure
#     operator. Measures the amount of edges present in the image.
#     :param img: the image the measure is applied to
#     :type img: numpy.ndarray
#     :returns: numpy.float32 -- the degree of focus
#     """
#     kernel = numpy.array([-1, 2, -1])
#     laplacianX = numpy.abs(cv2.filter2D(img, -1, kernel))
#     laplacianY = numpy.abs(cv2.filter2D(img, -1, kernel.T))
#     return numpy.mean(laplacianX + laplacianY)


# def TENG(img):
#     """Implements the Tenengrad (TENG) focus measure operator.
#     Based on the gradient of the image.
#     :param img: the image the measure is applied to
#     :type img: numpy.ndarray
#     :returns: numpy.float32 -- the degree of focus
#     """
#     gaussianX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
#     gaussianY = cv2.Sobel(img, cv2.CV_64F, 1, 0)
#     return numpy.mean(gaussianX * gaussianX +
#                       gaussianY * gaussianY)


# def MLOG(img):
#     """Implements the MLOG focus measure algorithm.
#     :param img: the image the measure is applied to
#     :type img: numpy.ndarray
#     :returns: numpy.float32 -- the degree of focus
#     """
#     return numpy.max(cv2.convertScaleAbs(cv2.Laplacian(img, 3)))