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
