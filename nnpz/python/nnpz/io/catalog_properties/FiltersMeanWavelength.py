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
Created: 12/04/18
Author: Alejandro Alvarez Ayllon
"""

from __future__ import division, print_function

import numpy as np

from nnpz.exceptions import UnknownNameException
from nnpz.io import CatalogReader


class FiltersMeanWavelength(CatalogReader.CatalogPropertyInterface):
    """Catalog property to retrieve the mean of the filter transmissions"""

    def __init__(self, filter_dict, nan_flags=None):
        """
        Constructor

        Args:
            filter_dict: A dictionary where the key is the band name, and the value the
                columns containing the filter transmission mean.
            nan_flags: A list of values which when found in the photometry
                columns are replaced with NaN.
        """
        self.__filter_dict = filter_dict
        self.__nan_flags = nan_flags if nan_flags else []

    def __call__(self, catalog):
        """
        Returns the filters transmission means of the catalog for each entry.

        Args:
            catalog: The catalog to read the filter transmission means from

        Returns:
            A dictionary where the key is the band name, and the value a
            numpy array of single precision floats where the
            with as many entries as the target catalog has.

        Raises:
            UnknownNameException: If the given file does misses any of the
            expected columns
        """
        for column_name in self.__filter_dict.values():
            if column_name not in catalog.colnames:
                raise UnknownNameException('Missing column {}'.format(column_name))

        data = {}
        for filter_name, column_name in self.__filter_dict.items():
            data[filter_name] = np.array(catalog[column_name], copy=True, dtype=np.float32)
            for flag in self.__nan_flags:
                data[filter_name][data == flag] = np.nan

        return data
