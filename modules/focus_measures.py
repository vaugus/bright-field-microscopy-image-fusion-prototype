import skimage
import numpy as np


class FocusMeasures(object):

    def __init__(self):
        pass

    def laplacian_filter(self, arr, kernel_size=3, square):
        gradients = skimage.filters.laplace(arr, kernel_size)
        
        if square:
            gradients *= gradients

        return gradients