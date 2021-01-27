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
Created on: 08/02/18
Author: Nikolaos Apostolakos
"""

from nnpz.weights import WeightPhotometryProvider


class CopiedPhotometry(WeightPhotometryProvider):
    """
    Dummy implementation of a WeightPhotometryProvider: The photometry used for the
    weighting are exactly those used for the K-NN search.
    """

    def __init__(self, ref_phot):
        self.__ref_phot = ref_phot

    def __call__(self, ref_i, cat_i, flags):
        return self.__ref_phot[ref_i]
