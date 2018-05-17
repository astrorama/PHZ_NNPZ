"""
Created on: 30/01/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np

from nnpz.exceptions import *
from nnpz.io import CatalogReader


class Photometry(CatalogReader.CatalogPropertyInterface):
    """Catalog property for retrieving the photometry at NNPZ format"""


    def __init__(self, column_list, nan_flags=[]):
        """Creates a new instance for the given column names.

        Args:
            column_list: A list of tuples with the columns containing the
                photometry. The first element of the tuple is the name of the
                value column and the second the error column.
            nan_flags: A list of values which when found in the photometry
                columns are replaced with NaN

        Note: The error columns are allowed to be None, in which case the errors
        of the band will be set to 0.
        """
        self.__column_list = column_list
        self.__nan_flags = nan_flags


    def __call__(self, catalog):
        """Returns the photometry of the catalog.

        Args:
            catalog: The catalog to read the photometry from

        Returns:
            A three dimensional numpy array of single precision floats where the
            first dimension has the same size as the number of objects the table
            contains photometries for, the second axis has same size as the
            given bands and the third axis has always size two, where the first
            element represents the photometry value and te second the
            uncertainty.

        Raises:
            UnknownNameException: If the given file does misses any of the
            expected columns
        """
        for value, error in self.__column_list:
            if not value in catalog.colnames:
                raise UnknownNameException('Missing column {}'.format(value))
            if not error is None and not error in catalog.colnames:
                raise UnknownNameException('Missing column {}'.format(error))

        # Construct the result array with zeros
        data = np.zeros((len(catalog), len(self.__column_list), 2), dtype=np.float32)

        # Populate the data
        for i, (value, error) in enumerate(self.__column_list):
            data[:,i,0] = catalog[value]
            if not error is None:
                data[:,i,1] = catalog[error]

        # Replace all the NaN flags
        for flag in self.__nan_flags:
            data[data==flag] = np.nan

        return data