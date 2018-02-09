"""
Created on: 08/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import abc


class WeightPhotometryProvider(object):
    __metaclass__ = abc.ABCMeta


    @abc.abstractmethod
    def __call__(self, ref_i, cat_i):
        return