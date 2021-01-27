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
Created on: 13/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np

from nnpz.photometry import PhotometryPrePostProcessorInterface


class PhotonPrePostProcessor(PhotometryPrePostProcessorInterface):
    """
    Pre/Post processor for producing photometry values in photon/cm^2/s
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
        pass

    def preProcess(self, sed):
        """Converts the given SED from ergs/cm^2/s/A to photon/cm^2/s/A.

        Args:
            sed: The SED to convert

        Returns:
            The SED expressed in photon/cm^2/s/A

        The conversion is taken from http://www.stsci.edu/~strolger/docs/UNITS.txt
        and is implemented as the equation:
        [Y photon/cm^2/s/A] = 5.03411250E+07 * [X1 erg/cm^2/s/A] * [X2 A]
        """
        result = np.ndarray(sed.shape, dtype=np.float32)
        result[:, 0] = sed[:, 0]
        result[:, 1] = 5.03411250E7 * sed[:, 1] * sed[:, 0]
        return result

    def postProcess(self, intensity, filter_name):
        """Returns the intensity unmodified. The filter info is ignored."""
        return intensity
