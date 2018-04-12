"""
Created: 12/04/18
Author: Alejandro Alvarez Ayllon
"""

from __future__ import division, print_function

import numpy as np

from nnpz.exceptions import UnknownNameException
from nnpz.io import CatalogReader


class FiltersTransmission(CatalogReader.CatalogPropertyInterface):
    """Catalog property to retrieve the mean of the filter transmissions"""

    def __init__(self, filter_dict, nan_flags=[]):
        """
        Constructor

        Args:
            filter_dict: A dictionary where the key is the band name, and the value the
                columns containing the filter transmission mean.
            nan_flags: A list of values which when found in the photometry
                columns are replaced with NaN.
        """
        self.__filter_dict = filter_dict
        self.__nan_flags = nan_flags

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
        for filter_name, column_name in self.__filter_dict.iteritems():
            data[filter_name] = np.array(catalog[column_name], copy=True, dtype=np.float32)
            for flag in self.__nan_flags:
                data[filter_name][data == flag] = np.nan

        return data
