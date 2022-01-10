#
# Copyright (C) 2012-2022 Euclid Science Ground Segment
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
Created on: 26/04/18
Author: Nikolaos Apostolakos
"""
from typing import Tuple

import numpy as np
from nnpz.weights.WeightCalculatorInterface import WeightCalculatorInterface


class InverseEuclideanWeight(WeightCalculatorInterface):
    """
    Compute the weight as the inverse of the Euclidean distance.
    For two identical points, this will be infinity since their distance is 0.
    This distance should only be used as a fall-back for whenever the likelihood of all
    neighbors become too small.
    """

    def __call__(self, ref_objs: np.ndarray, target_obj: np.ndarray) -> Tuple[np.ndarray, int]:
        val_1 = ref_objs[..., 0]
        val_2 = target_obj[..., 0, np.newaxis]

        distance = np.sqrt(np.sum((val_1 - val_2) * (val_1 - val_2), axis=0))
        return 1. / distance
