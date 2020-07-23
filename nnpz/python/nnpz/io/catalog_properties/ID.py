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

from nnpz.exceptions import UnknownNameException
from nnpz.io import CatalogReader


class ID(CatalogReader.CatalogPropertyInterface):
    """Catalog property for retrieving the IDs of the objects"""

    def __init__(self, col_name='ID'):
        """Creates a new instance with the given column name"""
        self.__col_name = col_name

    def __call__(self, catalog):
        """Returns the IDs of the objects.

        Args:
            catalog: The catalog to read the IDs from

        Returns:
            A numpy array of long integers with the IDs

        Raises:
            UnknownNameException: If the catalog does not contain the ID column
        """
        if self.__col_name not in catalog.colnames:
            raise UnknownNameException('Missing column {}'.format(self.__col_name))
        return np.asarray(catalog[self.__col_name], dtype=np.int64)
