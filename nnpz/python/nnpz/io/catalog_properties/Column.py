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
Created on: 30/01/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np

from nnpz.exceptions import *
from nnpz.io import CatalogReader


class Column(CatalogReader.CatalogPropertyInterface):
    """Catalog property for retrieving a column of the catalog"""


    def __init__(self, col_name, dtype=np.float32):
        """Creates a new instance with the given column name"""
        self.__col_name = col_name
        self.__dtype = dtype


    def __call__(self, catalog):
        """Returns the values of the column.

        Args:
            catalog: The catalog to read the column from

        Returns:
            A numpy array of type as the one defined at the constructor

        Raises:
            UnknownNameException: If the catalog does not contain the column
        """
        if not self.__col_name in catalog.colnames:
            raise UnknownNameException('Missing column {}'.format(self.__col_name))
        return np.asarray(catalog[self.__col_name], dtype=self.__dtype)
