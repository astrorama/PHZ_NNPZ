from __future__ import division, print_function

import numpy as np
from nnpz.neighbor_selection import BruteForceSelector


class AngularDistance(BruteForceSelector.DistanceMethodInterface):
    """
    Angular distance
    """

    def __call__(self, ref_data_values, ref_data_errors, coord_values, coord_errors):
        angular_num = np.dot(ref_data_values, coord_values)
        angular_den = np.linalg.norm(ref_data_values, axis=1) * np.linalg.norm(coord_values)
        angular = angular_num / angular_den

        return np.sin(np.arccos(angular))
