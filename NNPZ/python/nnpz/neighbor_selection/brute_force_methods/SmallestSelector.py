"""
Created on: 18/01/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np

from nnpz.neighbor_selection import BruteForceSelector


class SmallestSelector(BruteForceSelector.SelectionMethodInterface):
    """SelectionMethodInterface which selects the n smallest values"""

    def __init__(self, count):
        """Creates a new SmallestSelector

        Args:
            count: The number of values to select
        """
        self.__count = count

    def __call__(self, distances):
        """Returns the indices of the smallest distances.

        For argument and return description see the interface documentation.
        """

        # The argpartition method of numpy gives the n smallest chi2 values
        # without doing a full sorting, so it's faster (but the results are not
        # themselves ordered, which we don't care). The copy is necessary because
        # otherise the huge objects of the argpartition results will stay alive
        # in memory because the underlying data are shared and the memory will
        # be filled very fast.
        return np.copy(distances.argpartition(self.__count)[:self.__count])
