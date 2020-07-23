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
