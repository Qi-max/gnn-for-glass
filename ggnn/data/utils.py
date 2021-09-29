import os
import numpy as np
import pandas as pd


def list_like():
    return (list, np.ndarray, tuple, pd.Series)

def check_output_path(output_path, msg="output path", exist_ok=False):
    if os.path.exists(output_path):
        print("{}: {} already exists!".format(msg, output_path))
    else:
        os.makedirs(output_path, exist_ok=exist_ok)
        print("create {}: {} successful!".format(msg, output_path))


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.
    Inspired by Cgcnn, "https://github.com/txie-93/cgcnn"
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Args:
            dmin (float): Minimum interatomic distance
            dmax (float): Maximum interatomic distance
            step (float): Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        self.var = var if var else step

    def expand(self, distances):
        """
        Apply Gaussian distance filter to a distance array
        Args:
            distances (Array, (N)): A distance array

        Returns:
            expanded_distance (Array, (N, len(self.filter)): Expanded distance
                array using the Gaussian filter.
        """
        return np.exp(-(np.expand_dims(distances, axis=-1) - self.filter)**2 /
                      self.var**2)
