from __future__ import division, print_function

import numpy as np

from scipy.optimize import newton


# Chi2
def chi2(a, refval, referr, coordval, coorderr):
    nom = (refval * a[:, np.newaxis] - coordval) ** 2
    den = (referr * a[:, np.newaxis]) ** 2 + coorderr ** 2
    return np.sum(nom / den, axis=1)


class Chi2Scaling(object):
    """Find the appropriate scaling minimizing the chi2 and applying a prior"""

    def __init__(self, prior, a_min=1e-5, a_max=1e5, n_samples=1000,
                 batch_size=1000, max_iter=20, rtol=1e-4, epsilon=1e-4):
        """
        Constructor
        Args:
            prior: A function that models the prior for the scale factor a
            a_min: Minimum acceptable value for the scale
            a_max: Maximum acceptable value for the scale
            n_samples: Number of samples to use for finding the edges of the prior
            batch_size: Limit the minimization to this number of closest points
            max_iter: Maximum number of iterations for the minimizer
            rtol: Relative tolerance for the stop condition
            epsilon: Epsilon used for the numeric derivative
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
        self.__batch_size = batch_size
        self.__max_iter = max_iter
        self.__rtol = rtol
        self.__epsilon = epsilon

    def __call__(self, ref_values, ref_errors, coord_values, coord_errors):
        """
        Minimizes the chi2 distance using the scale factor a, which is constrained by
        the prior passed to the constructor
        """

        # Target function to be minimized
        def chi2_prior(a, *args):
            return chi2(a, *args) - np.log(self.prior(a))

        # Use newtown method to optimize all within in one go
        # Implies looking for a 0 on the derivative
        def chi2_prior_da(a, *args):
            return (chi2_prior(a + self.__epsilon, *args) - chi2_prior(a - self.__epsilon, *args)) / self.__epsilon * 2

        # Do an informed guess
        nom = ref_values * coord_values / coord_errors ** 2
        den = ref_values ** 2 / coord_errors ** 2
        a = np.sum(nom, axis=1) / np.sum(den, axis=1)

        # Clip those outside the bounds
        ge_min = a >= self.a_min
        le_max = a <= self.a_max
        a[np.logical_not(ge_min)] = self.a_min
        a[np.logical_not(le_max)] = self.a_max

        reference_mask = np.logical_and(ge_min, le_max)

        if reference_mask.any():
            # Compute chi2 with the guessed scaling
            distances = np.full(len(reference_mask), np.inf)
            distances[reference_mask] = chi2(
                a[reference_mask], ref_values[reference_mask, :], ref_errors[reference_mask, :], coord_values,
                coord_errors
            )

            # Prune objects that are far away
            prune_idx = np.argsort(distances)[self.__batch_size:]
            reference_mask[prune_idx] = False

            try:
                new_a = newton(
                    chi2_prior_da, a[reference_mask], args=(
                        ref_values[reference_mask, :], ref_errors[reference_mask, :], coord_values, coord_errors
                    ),
                    maxiter=self.__max_iter, rtol=self.__rtol, disp=False
                )
                not_nan = np.isnan(new_a) == False
                reference_mask[reference_mask] = not_nan
                a[reference_mask] = new_a[not_nan]
            except RuntimeError:
                pass

        return a
