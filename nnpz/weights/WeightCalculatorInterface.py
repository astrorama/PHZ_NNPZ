"""
Created on: 09/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import abc


class WeightCalculatorInterface(object):
    __metaclass__ = abc.ABCMeta


    @abc.abstractmethod
    def __call__(self, obj_1, obj_2, flags):
        return