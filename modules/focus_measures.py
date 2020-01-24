import abc

import skimage
import numpy as np


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
        gradients = skimage.filters.laplace(arr, kernel_size)

        if square:
            gradients *= gradients

        return gradients

    def execute(self, **kwargs):
        dataset = kwargs['dataset']
        gray_dataset = kwargs['gray_dataset']
        
        # compute the edges of the images with the laplacian filter
        edges = np.stack([self.laplacian_filter(arr, 3, square=True)
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