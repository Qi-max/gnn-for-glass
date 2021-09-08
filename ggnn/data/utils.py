import os
import torch
import numpy as np
import pandas as pd


def list_like():
    return (list, np.ndarray, tuple, pd.Series)


def padding_tensor(features, neighbor_len, batch_len, device):
    if features.size(1) < neighbor_len:
        padding_tensor = torch.zeros(
            batch_len, neighbor_len - features.size(1), features.size(2)).to(device)
        return torch.cat([features, padding_tensor], 1)
    return features


def check_output_path(output_path, msg="output path", exist_ok=False):
    if os.path.exists(output_path):
        print("{}: {} already exists!".format(msg, output_path))
    else:
        os.makedirs(output_path, exist_ok=exist_ok)
        print("create {}: {} successful!".format(msg, output_path))


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom

    Inspired by Cgcnn, "https://github.com/txie-93/cgcnn"
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(np.expand_dims(distances, axis=-1) - self.filter)**2 /
                      self.var**2)
