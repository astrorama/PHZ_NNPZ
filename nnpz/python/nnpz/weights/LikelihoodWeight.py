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
Created on: 09/02/18
Author: Nikolaos Apostolakos
"""
from typing import Tuple

import numpy as np
from nnpz.flags.NnpzFlag import NnpzFlag
from nnpz.weights.WeightCalculatorInterface import WeightCalculatorInterface


class LikelihoodWeight(WeightCalculatorInterface):
    """
    Compute the weight as the likelihood of the chi2: $L = e^{-\\chi^2/2}$
    Note that the maximum weight can be 1 (when $\\chi^2 == 0$), and it gets
    asymptotically close to 0 as $\\chi^2$ grows.
    """

    def __call__(self, ref_objs: np.ndarray, target_obj: np.ndarray) -> Tuple[np.ndarray, int]:
        val_1 = ref_objs[..., 0]
        err_1 = ref_objs[..., 1]
        val_2 = target_obj[..., 0, np.newaxis]
        err_2 = target_obj[..., 1, np.newaxis]

        nom = ((val_1 - val_2) * (val_1 - val_2))
        den = ((err_1 * err_1) + (err_2 * err_2))

        chi2 = np.sum(nom / den, axis=0)
        return np.exp(-0.5 * chi2), NnpzFlag.Empty
