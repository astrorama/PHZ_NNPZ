"""
Created on: 26/04/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np

from nnpz.weights import WeightCalculatorInterface


class InverseChi2Weight(WeightCalculatorInterface):

    def __call__(self, obj_1, obj_2, flags):
        v1 = obj_1[:, 0]
        e1 = obj_1[:, 1]
        v2 = obj_2[:, 0]
        e2 = obj_2[:, 1]

        chi2 = np.sum(((v1 - v2) * (v1 - v2)) / ((e1 * e1) + (e2 * e2)))
        return 1. / chi2