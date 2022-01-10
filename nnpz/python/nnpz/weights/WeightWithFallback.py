#
#  Copyright (C) 2022 Euclid Science Ground Segment
#
#  This library is free software; you can redistribute it and/or modify it under the terms of
#  the GNU Lesser General Public License as published by the Free Software Foundation;
#  either version 3.0 of the License, or (at your option) any later version.
#
#  This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
#  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License along with this library;
#  if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301 USA
#
from typing import Tuple

import numpy as np
from nnpz.flags import NnpzFlag
from nnpz.weights.WeightCalculatorInterface import WeightCalculatorInterface


class WeightWithFallback(WeightCalculatorInterface):
    """
    Wrap to weight calculators, so if the first returns all weights 0 (i.e. all neighbors
    are too far), then it calls the second and sets a flag

    Args:
        calculator:
        fallback:
    """

    def __init__(self, calculator: WeightCalculatorInterface, fallback: WeightCalculatorInterface):
        self.__calculator = calculator
        self.__fallback = fallback
        self.__min_weight = np.finfo(np.float32).eps

    def __call__(self, ref_objs: np.ndarray, target_obj: np.ndarray) -> Tuple[np.ndarray, int]:
        weight, flags = self.__calculator(ref_objs, target_obj)
        if np.all(weight[np.isfinite(weight)] <= self.__min_weight):
            weight, flags2 = self.__fallback(ref_objs, target_obj)
            flags |= NnpzFlag.AlternativeWeightFlag | flags2
        return weight, flags
