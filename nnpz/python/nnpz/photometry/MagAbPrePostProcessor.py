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

import math

from nnpz.photometry import PhotometryPrePostProcessorInterface, FnuPrePostProcessor


class MagAbPrePostProcessor(PhotometryPrePostProcessorInterface):
    """Pre/Post processor for producing AB magnitudes.

    This processor is a wrapper around the FnuPrePostProcessor, which simply
    converts the output to AB magnitude.
    """

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
        self.__fnu = FnuPrePostProcessor(transmissions)

    def preProcess(self, sed):
        """Redirects the call to the FnuPrePostProcessor"""
        return self.__fnu.preProcess(sed)

    def postProcess(self, intensity, filter_name):
        """Returns the flux in AB magnitude.

        This method uses first the FnuPrePostProcessor to compute the flux in
        erg/s/cm^2/Hz. Then it converts the result to AB magnitude based on the
        equation from https://en.wikipedia.org/wiki/AB_magnitude:

        m_AB = -5/2 * log10(F_nu) - 48.6
        """
        flux = self.__fnu.postProcess(intensity, filter_name)
        return -2.5 * math.log10(flux) - 48.6
