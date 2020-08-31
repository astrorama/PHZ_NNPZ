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

from __future__ import division, print_function

import numpy as np
from nnpz.neighbor_selection import BruteForceSelector


class DirectedDistance(BruteForceSelector.DistanceMethodInterface):
    """
    Directed distance implementation
    See https://arxiv.org/pdf/1511.07623.pdf
    """

    def __call__(self, ref_data_values, ref_data_errors, coord_values, coord_errors):
        euclidean = ref_data_values - coord_values
        euclidean = euclidean * euclidean
        euclidean = np.sum(euclidean, axis=1)

        angular_num = np.dot(ref_data_values, coord_values)
        angular_den = np.linalg.norm(ref_data_values, axis=1) * np.linalg.norm(coord_values)
        angular = angular_num / angular_den

        return euclidean * np.sin(np.arccos(angular)) ** 2