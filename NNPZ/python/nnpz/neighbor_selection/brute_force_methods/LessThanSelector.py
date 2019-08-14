"""
Created on: 18/01/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np

from nnpz.neighbor_selection import BruteForceSelector


class LessThanSelector(BruteForceSelector.SelectionMethodInterface):
    """SelectionMethodInterface which selects all distances less than a value"""

    def __init__(self, trigger):
        """Creates a new LessThanSelector

        Args:
            trigger: The value to select distances less than
        """
        self.__trigger = trigger

    def __call__(self, distances):
        """Returns the indices of the distances smaller than the trigger.

        For argument and return description see the interface documentation.
        """

        return np.where(distances < self.__trigger)[0]
