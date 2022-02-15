#
# Copyright (C) 2012-2022 Euclid Science Ground Segment
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
Created on: 04/12/17
Author: Nikolaos Apostolakos
"""

import abc
from typing import List

import numpy as np


class FilterProviderInterface(metaclass=abc.ABCMeta):
    """Interface providing the filter transmissions"""

    @abc.abstractmethod
    def get_filter_names(self) -> List[str]:
        """
        Provides a list with the names of the filters.

        Returns:
            A python list with the names of the filters as strings
        """
        return

    @abc.abstractmethod
    def get_filter_transmission(self, name) -> np.ndarray:
        """
        Provides the transmission curve of the filter with the given name.

        Args:
            name: The name of the filter to get the transmission for

        Returns:
            A 2D numpy array of single precision floating point numbers. The
            first dimension represents the knots of the filter transmission
            and the second one has always size 2, representing the wavelength
            (expressed in Angstrom) and the transmission value (in the range
            [0,1]).

        Raises:
            KeyError: If there is no filter with the given name
        """
        return
