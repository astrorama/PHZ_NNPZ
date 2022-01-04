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


import abc


class WeightPhotometryProvider(object):
    """
    Interface definition for methods that provide a new photometry to the weight methods
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, ref_i, cat_i, flags):
        """
        Return the photometry that must be used to compare the reference ref_i with the target
        cat_i. This may be the original photometry of the reference, or it may be a transformed
        or recomputed photometry that brings it to the target color space
        See Also:
            RecomputedPhotometry
            CopiedPhotometry

        Args:
            ref_i: Reference index
            cat_i: Target index
            flags: NnpzFlag

        Returns:
            A structured array with the filter names as attributes, and one dimension with two
            positions: value and error
        """
        return
