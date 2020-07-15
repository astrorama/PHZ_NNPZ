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
Created on: 13/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np

from nnpz.photometry import PhotometryPrePostProcessorInterface


class PhotonPrePostProcessor(PhotometryPrePostProcessorInterface):
    """Pre/Post processor for producing photometry values in photon/cm^2/s"""

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

    def postProcess(self, intensity, filter_name, filter_trans):
        """Returns the intensity unmodified. The filter info is ignored."""
        return intensity
