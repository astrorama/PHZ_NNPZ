"""
Created on: 30/01/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np

from nnpz.exceptions import *
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
        if not self.__col_name in catalog.colnames:
            raise UnknownNameException('Missing column {}'.format(self.__col_name))
        return np.asarray(catalog[self.__col_name], dtype=np.int64)
