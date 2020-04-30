from __future__ import division, print_function

import numpy as np

from nnpz.neighbor_selection import BruteForceSelector
from scipy.optimize import minimize_scalar


class ScaledChi2Distance(BruteForceSelector.DistanceMethodInterface):
    """Chi2 distance implementation"""

    def __init__(self, scaling_method):
        """
        Constructor
        Args:
            scaling_method: Scaling method to apply
        """
        assert hasattr(scaling_method, '__call__')
        self.__scaling_method = scaling_method

    def __call__(self, ref_values, ref_errors, coord_values, coord_errors):
        """
        Minimizes the chi2 distance using the scale factor a, which is constrained by
        the prior passed to the constructor
        """
        # Scaling
        a = self.__scaling_method(ref_values, ref_errors, coord_values, coord_errors)
        # Distances
        nom = (a[:, np.newaxis] * ref_values - coord_values) ** 2
        den = (a[:, np.newaxis] * ref_errors) ** 2 + coord_errors ** 2
        d = np.sum(nom / den, axis=1)
        return d, a
