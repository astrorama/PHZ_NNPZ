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
Created on: 19/01/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np

from nnpz.neighbor_selection import BruteForceSelector


class EuclideanDistance(BruteForceSelector.DistanceMethodInterface):
    """Euclidean distance implementation

    WARNING: The Euclidean distance ignores the uncertainties
    """

    def __call__(self, ref_data_values, ref_data_errors, coord_values, coord_errors):
        """Returns the euclidean distances.

        For argument and return description see the interface documentation.

        WARNING: The Euclidean distance ignores the uncertainties
        """

        dist = ref_data_values - coord_values
        dist = dist * dist
        dist = np.sum(dist, axis=1)
        dist = np.sqrt(dist)

        return dist
