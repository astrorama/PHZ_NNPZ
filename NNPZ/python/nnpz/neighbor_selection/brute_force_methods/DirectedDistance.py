from __future__ import division, print_function

import numpy as np
from nnpz.neighbor_selection import BruteForceSelector


class DirectedDistance(BruteForceSelector.DistanceMethodInterface):
    """
    Directed distance implementation
    See https://arxiv.org/pdf/1511.07623.pdf
    """

    def __call__(self, ref_data_values, ref_data_errors, coord_values, coord_errors):
        euclidean = ref_data_values - coord_values
        euclidean = euclidean * euclidean
        euclidean = np.sum(euclidean, axis=1)

        angular_num = np.dot(ref_data_values, coord_values)
        angular_den = np.linalg.norm(ref_data_values, axis=1) * np.linalg.norm(coord_values)
        angular = angular_num / angular_den

        return euclidean * np.sin(np.arccos(angular)) ** 2
