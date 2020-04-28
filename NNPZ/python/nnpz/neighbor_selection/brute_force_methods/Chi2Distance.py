"""
Created on: 19/01/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np

from nnpz.neighbor_selection import BruteForceSelector


class Chi2Distance(BruteForceSelector.DistanceMethodInterface):
    """Chi2 distance implementation"""

    def __call__(self, ref_data_values, ref_data_errors, coord_values, coord_errors):
        """Returns the chi2 distances.

        For argument and return description see the interface documentation.

        The chi2 or each ref_data entry is computed by  summing for all the
        dimensions of the parameter space the terms
        (V_ref - V_coord)^2 / (E_ref^2 + E_coord^2)
        """

        nom = ref_data_values - coord_values
        nom = nom * nom

        den = ref_data_errors * ref_data_errors + coord_errors * coord_errors

        return np.sum(nom / den, axis=1), np.ones(len(ref_data_values))
