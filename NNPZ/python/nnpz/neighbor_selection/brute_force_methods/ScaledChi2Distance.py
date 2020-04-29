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

    def __call__(self, ref_data_values, ref_data_errors, coord_values, coord_errors):
        """
        Minimizes the chi2 distance using the scale factor a, which is constrained by
        the prior passed to the constructor
        """

        d = np.ones(len(ref_data_values))
        a = np.ones(len(ref_data_values))

        for i in range(len(a)):
            # Do an educated guess ignoring the errors
            ri = ref_data_values[i, :]
            fi = coord_values
            si = coord_errors
            sri = ref_data_errors[i, :]

            def chi2(a):
                nom = (a * ri[:, np.newaxis] - fi[:, np.newaxis]) ** 2
                den = (a * sri[:, np.newaxis]) ** 2 + si[:, np.newaxis] ** 2
                return np.sum(nom / den, axis=0)

            def likelihood(a):
                return np.exp(-chi2(a) / 2)

            # Do an informed guess
            a_guess = np.sum(ri * fi / si ** 2) / np.sum(ri ** 2 / si ** 2)

            # If the guess is out of bounds, clip and finish
            if a_guess <= self.__a_min:
                a[i] = self.__a_min
            elif a_guess >= self.__a_max:
                a[i] = self.__a_max
            # If it is within, do the maximization of the likelihood * prior
            else:
                # Which is done minimizing the negative
                a[i] = fmin(lambda a: -self.__prior(a) * likelihood(a), a[i], disp=False)[0]
            # Compute the chi2 distance for the given a
            d[i] = chi2(a[i])

        return d, a
