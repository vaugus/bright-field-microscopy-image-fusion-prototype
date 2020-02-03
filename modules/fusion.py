from modules.pre_processing import PreProcessing
from modules.focus_measures import EnergyOfLaplacian

import os

import numpy as np
from PIL import Image
from SSIM_PIL import compare_ssim
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR


class Fusion(object):

    def __init__(self):
        super().__init__()
        self.pre_processing = PreProcessing()
        self.energy_of_laplacian = EnergyOfLaplacian()

    def run(self, path):
        """Method and function names are lower_case_with_underscores.

        Always use self as first arg.
        """
        # open the dataset images
        size = (1280, 960)
        # size = (640, 480)
        # size = (320, 240)
        # size = (160, 120)
        # size = (80, 60)
        # size = (40, 30)
        # size = (20, 15)


        # size = (544, 544)
        # size = (272, 272)
        # size = (136, 136)
        # size = (68, 68)
        # size = None


        dataset = self.pre_processing.open_dataset(path, size)

        # convert images to grayscale
        gray_dataset = self.pre_processing.grayscale_dataset(dataset, 'luminance')

        result = self.energy_of_laplacian.execute(
            dataset=dataset, gray_dataset=gray_dataset)

        A = self.pre_processing.ndarray_to_image(result)

        mssim = 0.0
        psnr = 0.0

        arrs = []
        for elem in dataset:
            tmp = self.pre_processing.image_to_ndarray(elem)
            arrs.append((tmp * 255.).astype(np.uint8))

        for i in range(len(dataset)):
            mssim += compare_ssim(dataset[i], A)
            tmp = PSNR(arrs[i], (result * 255.).astype(np.uint8))
            psnr += tmp

        print(mssim / len(dataset))
        print(psnr / len(dataset))

        A.show(title="A")
