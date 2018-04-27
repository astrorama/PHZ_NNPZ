"""
Created on: 26/04/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np

from nnpz.weights import WeightCalculatorInterface


class InverseEuclideanWeight(WeightCalculatorInterface):

    def __call__(self, obj_1, obj_2, flags):
        v1 = obj_1[:, 0]
        v2 = obj_2[:, 0]

        distance = np.sqrt(np.sum((v1 - v2) * (v1 - v2)))
        return 1. / distance