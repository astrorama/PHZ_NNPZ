#
# Copyright (C) 2012-2021 Euclid Science Ground Segment
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
Created on: 14/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from typing import Dict

import numpy as np
from astropy import constants, units as u

from .photometry_processor_interface import PhotometryPrePostProcessorInterface

C = constants.c.to(u.Angstrom / u.s).value


class FnuuJyPrePostProcessor(PhotometryPrePostProcessorInterface):
    """
    Pre/Post processor for producing  photometry in uJy.

    Args:
        transmissions: A dictionary where the key is the filter name, and the value
            the filter transmission as a 2D numpy array of single
            precision floating point numbers. The first dimension represents
            the knots of the filter transmission and the second one has
            always size 2, representing the wavelength (expressed in
            Angstrom) and the transmission value (in the range [0,1]).
    """

    def __init__(self, transmissions: Dict[str, np.ndarray]):
        self.__filter_norm = {}
        # Pre-compute the filter norm here
        for filter_name, filter_trans in transmissions.items():
            lambda_gt_0 = filter_trans[:, 0] > 0.
            if np.any(filter_trans[~lambda_gt_0, 1] > 0):
                raise ValueError(
                    'There is a transmission value for a lambda less than or equal to 0!')
            lambd = filter_trans[:, 0][lambda_gt_0]
            norm_f = filter_trans[:, 1][lambda_gt_0] / lambd
            self.__filter_norm[filter_name] = C * np.trapz(norm_f, x=lambd)

    def preProcess(self, sed: np.ndarray) -> np.ndarray:
        """
        Multiply the SED with the wavelength
        """
        res = sed.copy()
        res[:, 1] = res[:, 1] * res[:, 0]
        return res

    def postProcess(self, intensity: float, filter_name: str) -> float:
        """
        Converts the intensity to flux density by normalizing for the filter.

        Args:
            intensity: The intensity
            filter_name: The filter name

        Returns:
            The flux density in uJy

        The given intensity is normalized for the filter as described in
        http://www.asvo.org.au/blog/2013/03/21/derivation-of-ab-magnitudes/ .
        According this recipe, the filter transmission is first multiplied with
        c/lambda^2 (to compensate for the fact that the integration of the
        filter is done in wavelength instead of frequency). Then the total
        integral of the result is used to normalize the intensity.

        The filter integration is performed using the trapezoidal rule.
        """
        return (intensity / self.__filter_norm[filter_name]) * 1E29
