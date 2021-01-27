#
# Copyright (C) 2012-2021 Euclid Science Ground Segment
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
