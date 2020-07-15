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
Created on: 13/07/18
Author: Alejandro Alvarez Ayllon
"""

from __future__ import division, print_function

import numpy as np

from nnpz.exceptions import *
from nnpz.io import CatalogReader


class EBV(CatalogReader.CatalogPropertyInterface):
    """Catalog property to retrieve the galactic extinction"""


    def __init__(self, ebv_column, nan_flags=[]):
        """Creates a new instance for the given column names.

        Args:
            ebv_column: The name of the column that contains EBV
            nan_flags: A list of values which when found are replaced with NaN

        """
        self.__ebv_column = ebv_column
        self.__nan_flags = nan_flags


    def __call__(self, catalog):
        """Returns the values of the extinction

        Args:
            catalog: The catalog to read the photometry from

        Returns:
            A one dimensional numpy array of single precision floats with the
            same size as the number of objects the table contains. It represents
            the E(B-V) value.

        Raises:
            UnknownNameException: If the given file does misses any of the
            expected columns
        """
        if self.__ebv_column not in catalog.colnames:
            raise UnknownNameException('Missing column {}'.format(self.__ebv_column))

        data = catalog[self.__ebv_column]
        for flag in self.__nan_flags:
            data[data == flag] = np.nan

        return data
