from __future__ import division, print_function

import numpy as np

from nnpz.neighbor_selection import BruteForceSelector
from scipy.optimize import minimize_scalar


class ScaledChi2Distance(BruteForceSelector.DistanceMethodInterface):
    """Chi2 distance implementation"""

    def __init__(self, prior, a_min=1e-5, a_max=1e2, n_samples=1000, max_iter=20, xtol=1e-4):
        """
        Constructor
        Args:
            prior: A function that models the prior for the scale factor a
            a_min: Minimum acceptable value for the scale
            a_max: Maximum acceptable value for the scale
            n_samples: Number of samples to use for finding the edges of the prior
            max_iter: Maximum number of iterations for the minimizer
            xtol: Relative tolerance for the minimizer
        """
        assert hasattr(prior, '__call__')
        self.prior = prior
        # Find the min and max from the prior itself
        self.__a_s = np.sort(np.append(np.arange(1, np.floor(a_max) + 1),
                                       np.exp(np.linspace(np.log(a_min), np.log(a_max), n_samples))))
        pv = prior(self.__a_s)
        gt0 = pv > 0.
        self.a_min = self.__a_s[np.argmax(gt0)]
        self.a_max = np.flip(self.__a_s)[np.argmax(np.flip(gt0))]
        self.__max_iter = max_iter
        self.__xtol = xtol

    def __call__(self, ref_values, ref_errors, coord_values, coord_errors):
        """
        Minimizes the chi2 distance using the scale factor a, which is constrained by
        the prior passed to the constructor
        """

        # Do an informed guess
        num = ref_values * coord_values / coord_errors ** 2
        den = ref_values ** 2 / coord_errors ** 2
        a = np.sum(num, axis=1) / np.sum(den, axis=1)

        # Clip those outside the bounds
        ge_min = a >= self.a_min
        le_max = a <= self.a_max
        a[np.logical_not(ge_min)] = self.a_min
        a[np.logical_not(le_max)] = self.a_max

        # Try improving those within
        within_mask = np.logical_and(ge_min, le_max)
        within = np.arange(len(a))[within_mask]

        for i in within:
            # Do an educated guess ignoring the errors
            ri = ref_values[i, :]
            sri = ref_errors[i, :]
            fi = coord_values
            si = coord_errors

            def chi2(a):
                nom = (a * ri[:, np.newaxis] - fi[:, np.newaxis]) ** 2
                den = (a * sri[:, np.newaxis]) ** 2 + si[:, np.newaxis] ** 2
                return np.sum(nom / den)

            def likelihood(a):
                return np.exp(-chi2(a) / 2)

            # Get a few samples around the guess
            ai = np.argmax(self.__a_s > a[i])
            a0 = self.__a_s[max(0, ai - 1)]
            a1 = self.__a_s[min(len(self.__a_s) - 1, ai + 1)]
            a[i] = minimize_scalar(lambda a: chi2(a) / 2 - np.log(self.prior(a)), bracket=(a0, a1), method='brent',
                                   options=dict(xtol=self.__xtol, maxiter=self.__max_iter)).x
            # The optimizer can move us outside the limits!
            a[i] = max(min(self.a_max, a[i]), self.a_min)

        # Compute the final distances
        nom = (a[:, np.newaxis] * ref_values - coord_values) ** 2
        den = (a[:, np.newaxis] * ref_errors) ** 2 + coord_errors ** 2
        d = np.sum(nom / den, axis=1)
        return d, a
