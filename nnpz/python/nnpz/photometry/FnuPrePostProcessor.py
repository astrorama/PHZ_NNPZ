#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
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

import numpy as np

from nnpz.photometry import PhotometryPrePostProcessorInterface

# The speed of light in Angstrom/s
C = 299792458E10


class FnuPrePostProcessor(PhotometryPrePostProcessorInterface):
    """Pre/Post processor for producing  photometry in erg/s/cm^2/Hz"""

    def __init__(self):
        self.__filter_norm = {}

    def preProcess(self, sed):
        """Multiply the SED with the wavelength"""
        res = sed.copy()
        res[:, 1] = res[:, 1] * res[:, 0]
        return res

    def postProcess(self, intensity, filter_name, filter_trans):
        """Converts the intensity to flux density by normalizing for the filter.

        Args:
            intensity: The intensity
            filter_name: The filter name
            filter_trans: The filter transmission

        Returns:
            The flux density in erg/s/cm^2/Hz

        The given intensity is normalized for the filter as described in
        http://www.asvo.org.au/blog/2013/03/21/derivation-of-ab-magnitudes/ .
        According this recipe, the filter transmission is first multiplied with
        c/lambda^2 (to compensate for the fact that the integration of the
        filter is done in wavelength instead of frequency). Then the total
        integral of the result is used to normalize the intensity.

        The filter integration is performed using the trapezoidal rule.
        """

        # First get the filter normalization. We cache the factors to avoid
        # recomputing them when the processor is used for multiple SEDs.
        if filter_name not in self.__filter_norm:
            lambda_gt_0 = filter_trans[:, 0] > 0.
            if np.any(filter_trans[:, 1][~lambda_gt_0] != 0.):
                raise ValueError(
                    'There is a transmission value for a lambda less than or equal to 0!')
            lambd = filter_trans[:, 0][lambda_gt_0]
            norm_f = filter_trans[:, 1][lambda_gt_0] / lambd
            self.__filter_norm[filter_name] = C * np.trapz(norm_f, x=lambd)
        norm = self.__filter_norm[filter_name]

        return intensity / norm