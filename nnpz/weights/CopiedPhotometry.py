"""
Created on: 08/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from nnpz.weights import WeightPhotometryProvider


class CopiedPhotometry(WeightPhotometryProvider):

    def __init__(self, ref_phot):
        self.__ref_phot = ref_phot

    def __call__(self, ref_i, cat_i):
        return self.__ref_phot[ref_i]