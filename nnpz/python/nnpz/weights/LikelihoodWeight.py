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
Created on: 09/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np
import math

from nnpz.weights import WeightCalculatorInterface


class LikelihoodWeight(WeightCalculatorInterface):

    def __call__(self, obj_1, obj_2, flags):
        v1 = obj_1[:, 0]
        e1 = obj_1[:, 1]
        v2 = obj_2[:, 0]
        e2 = obj_2[:, 1]

        chi2 = np.sum(((v1 - v2) * (v1 - v2)) / ((e1 * e1) + (e2 * e2)))
        return math.exp(-0.5 * chi2)
