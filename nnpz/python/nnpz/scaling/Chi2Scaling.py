#
# Copyright (C) 2012-2022 Euclid Science Ground Segment
#
# This library is free software; you can redistribute it and/or modify it under the terms of
# the GNU Lesser General Public License as published by the Free Software Foundation;
# either version 3.0 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this library;
# if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301 USA
#

"""
Implements scaling based on the chi^2 distance:

A first approximation is computed directly, ignoring errors from the reference sample.
A minimization is done later, taking into account all errors, looking for the scale factor
that minimizes the chi^2 distance between reference an target object.
"""

import warnings
from typing import Callable

import numpy as np
from ElementsKernel import Logging
from scipy.optimize import newton

logger = Logging.getLogger(__name__)


# Chi2
def _chi2(a: float, refval: np.ndarray, referr: np.ndarray, coordval: np.ndarray,
          coorderr: np.ndarray) -> np.ndarray:
    """
    chi2 distance
    """
    nom = (refval * a[:, np.newaxis] - coordval) ** 2
    den = (referr * a[:, np.newaxis]) ** 2 + coorderr ** 2
    return np.sum(nom / den, axis=1)


class Chi2Scaling:
    """
    Find the appropriate scaling minimizing the chi2 and applying a prior

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

    def __init__(self, prior: Callable[[float], float], a_min: float = 1e-5, a_max: float = 1e5,
                 n_samples: int = 1000, batch_size: int = 1000, max_iter: int = 20,
                 rtol: float = 1e-4, epsilon: float = 1e-4):
        if not hasattr(prior, '__call__'):
            raise TypeError('prior must be callable')
        self.prior = prior
        # Find the min and max from the prior itself
        self.__a_s = np.sort(
            np.append(
                np.arange(1, np.floor(a_max) + 1),
                np.exp(np.linspace(np.log(a_min), np.log(a_max), n_samples))
            )
        )
        prior_values = prior(self.__a_s)
        gt0 = prior_values > 0.
        self.a_min = self.__a_s[np.argmax(gt0)]
        self.a_max = np.flip(self.__a_s)[np.argmax(np.flip(gt0))]
        self.__batch_size = batch_size
        self.__max_iter = max_iter
        self.__rtol = rtol
        self.__epsilon = epsilon

    def __call__(self, reference: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Minimizes the chi2 distance using the scale factor `a`, which is constrained by
        the prior passed to the constructor
        """
        ref_values = reference[..., 0]
        ref_errors = reference[..., 1]
        coord_values = target[..., 0]
        coord_errors = target[..., 1]

        # We expect no NaNs
        assert not np.any(np.isnan(coord_values))
        assert not np.any(np.isnan(coord_errors))

        # Target function to be minimized
        def chi2_prior(scale, *args):
            # pylint: disable=no-value-for-parameter
            return _chi2(scale, *args) / 2 - np.log(self.prior(scale))

        # Use Newton method to optimize all within in one go
        # Implies looking for a 0 on the derivative
        def chi2_prior_da(scale, *args):
            # pylint: disable=no-value-for-parameter
            prior_plus = chi2_prior(scale + self.__epsilon, *args)
            # pylint: disable=no-value-for-parameter
            prior_minus = chi2_prior(scale - self.__epsilon, *args)
            return (prior_plus - prior_minus) / self.__epsilon * 2

        # Do an informed guess
        nom = ref_values * coord_values / coord_errors ** 2
        den = ref_values ** 2 / coord_errors ** 2
        scale = np.sum(nom, axis=1) / np.sum(den, axis=1)

        # Clip those outside the bounds
        ge_min = scale >= self.a_min
        le_max = scale <= self.a_max
        scale[np.logical_not(ge_min)] = self.a_min
        scale[np.logical_not(le_max)] = self.a_max

        reference_mask = np.logical_and(ge_min, le_max)

        if reference_mask.any():
            # Compute chi2 with the guessed scaling
            distances = np.full(len(reference_mask), np.inf)
            distances[reference_mask] = _chi2(
                scale[reference_mask], ref_values[reference_mask, :], ref_errors[reference_mask, :],
                coord_values, coord_errors
            )

            # Prune objects that are far away
            prune_idx = np.argsort(distances)[self.__batch_size:]
            reference_mask[prune_idx] = False

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    new_scale = newton(
                        chi2_prior_da, scale[reference_mask], args=(
                            ref_values[reference_mask, :], ref_errors[reference_mask, :],
                            coord_values, coord_errors
                        ),
                        maxiter=self.__max_iter, rtol=self.__rtol, disp=False
                    )
                not_nan = np.isfinite(new_scale)
                reference_mask[reference_mask] = not_nan
                scale[reference_mask] = new_scale[not_nan]
            except RuntimeError as e:
                logger.warning(e)

        return scale
