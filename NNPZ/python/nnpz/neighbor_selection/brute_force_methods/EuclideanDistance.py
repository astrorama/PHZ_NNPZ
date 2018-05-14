"""
Created on: 19/01/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np

from nnpz.neighbor_selection import BruteForceSelector


class EuclideanDistance(BruteForceSelector.DistanceMethodInterface):
    """Euclidean distance implementation

    WARNING: The Euclidean distance ignores the uncertainties
    """

    def __call__(self, ref_data_values, ref_data_errors, coord_values, coord_errors):
        """Returns the euclidean distances.

        For argument and return description see the interface documentation.

        WARNING: The Euclidean distance ignores the uncertainties
        """

        dist = ref_data_values - coord_values
        dist = dist * dist
        dist = np.sum(dist, axis=1)
        dist = np.sqrt(dist)

        return dist