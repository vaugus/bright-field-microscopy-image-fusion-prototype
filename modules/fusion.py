from modules.pre_processing import PreProcessing
from modules.focus_measures import EnergyOfLaplacian
from modules.evaluation import


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
        dataset = self.pre_processing.open_dataset(path)

        # convert images to grayscale
        gray_dataset = [self.pre_processing.image_to_ndarray(
            self.pre_processing.grayscale_averaging(img)) for img in dataset]

        result = self.energy_of_laplacian.execute(
            dataset=dataset, gray_dataset=gray_dataset)

        self.pre_processing.ndarray_to_image(result).show()

    def evaluate()
