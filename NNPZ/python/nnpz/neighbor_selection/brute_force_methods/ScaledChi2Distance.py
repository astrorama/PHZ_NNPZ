from __future__ import division, print_function

import numpy as np

from nnpz.neighbor_selection import BruteForceSelector
from scipy.optimize import fmin


class ScaledChi2Distance(BruteForceSelector.DistanceMethodInterface):
    """Chi2 distance implementation"""

    def __init__(self, prior, a_min, a_max):
        """
        Constructor
        Args:
            prior: A function that models the prior for the scale factor a
            a_min: Minimum acceptable value for the scale
            a_max: Maximum acceptable value for the scale
        """
        assert hasattr(prior, '__call__')
        self.__prior = prior
        self.__a_min = a_min
        self.__a_max = a_max

    def __call__(self, ref_values, ref_errors, coord_values, coord_errors):
        """
        Minimizes the chi2 distance using the scale factor a, which is constrained by
        the prior passed to the constructor
        """

        # Do an informed guess
        a = np.sum(ref_values * coord_values / coord_errors ** 2, axis=1) / np.sum(ref_values ** 2 / coord_errors ** 2,
                                                                                   axis=1)

        # Treat each one individually
        for i in range(len(a)):
            # Do an educated guess ignoring the errors
            ri = ref_values[i, :]
            sri = ref_errors[i, :]
            fi = coord_values
            si = coord_errors

            def chi2(a):
                nom = (a * ri[:, np.newaxis] - fi[:, np.newaxis]) ** 2
                den = (a * sri[:, np.newaxis]) ** 2 + si[:, np.newaxis] ** 2
                return np.sum(nom / den, axis=0)

            def likelihood(a):
                return np.exp(-chi2(a) / 2)

            # If the guess is out of bounds, clip and finish
            if a[i] <= self.__a_min:
                a[i] = self.__a_min
            elif a[i] >= self.__a_max:
                a[i] = self.__a_max
            # If it is within, do the maximization of the likelihood * prior
            else:
                # Which is done minimizing the negative
                a[i] = fmin(lambda a: -self.__prior(a) * likelihood(a), a[i], disp=False, maxiter=10, ftol=1e-2)[0]

        # Compute the final distances
        nom = (a[:, np.newaxis] * ref_values - coord_values) ** 2
        den = (a[:, np.newaxis] * ref_errors) ** 2 + coord_errors ** 2
        d = np.sum(nom / den, axis=1)
        return d, a
