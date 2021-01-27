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
Created on: 18/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np

from nnpz.photometry import PhotometryPrePostProcessorInterface


class FlambdaPrePostProcessor(PhotometryPrePostProcessorInterface):
    """Pre/Post processor for producing  photometry in erg/s/cm^2/A"""

    def __init__(self, transmissions):
        """
        Initialize the pre-post processor.

        Args:
            transmissions: A dictionary where the key is the filter name, and the value
                the filter transmission as a 2D numpy array of single
                precision floating point numbers. The first dimension represents
                the knots of the filter transmission and the second one has
                always size 2, representing the wavelength (expressed in
                Angstrom) and the transmission value (in the range [0,1]).
        """
        self.__filter_norm = {}
        for filter_name, filter_trans in transmissions.items():
            self.__filter_norm[filter_name] = np.trapz(filter_trans[:, 1], x=filter_trans[:, 0])

    def preProcess(self, sed):
        """Returns the SED unmodified"""
        return sed

    def postProcess(self, intensity, filter_name):
        """Converts the intensity to flux density by normalizing for the filter.

        Args:
            intensity: The intensity
            filter_name: The filter name

        Returns:
            The flux density in erg/s/cm^2/A

        The given intensity is normalized using the total integral of the filter
        over the wavelength in Angstrom.

        The filter integration is performed using the trapezoidal rule.
        """
        return intensity / self.__filter_norm[filter_name]
