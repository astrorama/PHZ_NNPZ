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

from nnpz.photometry import PhotometryPrePostProcessorInterface, FnuPrePostProcessor


class FnuuJyPrePostProcessor(PhotometryPrePostProcessorInterface):
    """Pre/Post processor for producing  photometry in uJy.

    This processor is a wrapper around the FnuPrePostProcessor, which simply
    converts the output to uJy.
    """

    def __init__(self):
        self.__fnu = FnuPrePostProcessor()

    def preProcess(self, sed):
        """Redirects the call to the FnuPrePostProcessor"""
        return self.__fnu.preProcess(sed)

    def postProcess(self, intensity, filter_name, filter_trans):
        """Returns the flux in uJy.

        This method uses first the FnuPrePostProcessor to compute the flux in
        erg/s/cm^2/Hz. Then, this value is converted to Jansky by multiplying
        with 10^23 (https://en.wikipedia.org/wiki/Jansky). Then it is converted
        to uJy by multiplying with 10^6.
        """
        flux = self.__fnu.postProcess(intensity, filter_name, filter_trans)
        return flux * 1E29
