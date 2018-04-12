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


    def __init__(self, ebv_column, ebv_err_column, nan_flags=[]):
        """Creates a new instance for the given column names.

        Args:
            ebv_column: The name of the column that contains EBV
            ebv_err_column: The name of the column that contains the associated error
            nan_flags: A list of values which when found are replaced with NaN

        Note: The error column is allowed to be None, in which case the errors
        of the EBV will be set to 0.
        """
        self.__ebv_column = ebv_column
        self.__ebv_err_column = ebv_err_column
        self.__nan_flags = nan_flags


    def __call__(self, catalog):
        """Returns the values of the extinction

        Args:
            catalog: The catalog to read the photometry from

        Returns:
            A two dimensional numpy array of single precision floats where the
            first dimension has the same size as the number of objects the table contains.
            The second axis has always the size two, where the first
            element represents the E(B-V) value and the second the
            error.

        Raises:
            UnknownNameException: If the given file does misses any of the
            expected columns
        """
        if self.__ebv_column not in catalog.colnames:
            raise UnknownNameException('Missing column {}'.format(self.__ebv_column))
        if self.__ebv_err_column is not None and self.__ebv_column not in catalog.colnames:
            raise UnknownNameException('Missing column {}'.format(self.__ebv_err_column))

        data = np.zeros((len(catalog), 2), dtype=np.float32)

        data[:, 0] = catalog[self.__ebv_column]
        if self.__ebv_err_column is not None:
            data[:, 1] = catalog[self.__ebv_err_column]

        for flag in self.__nan_flags:
            data[data == flag] = np.nan

        return data
