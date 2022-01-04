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
Created on: 30/01/18
Author: Nikolaos Apostolakos
"""


import numpy as np

from nnpz.exceptions import UnknownNameException
from nnpz.io import CatalogReader


class Photometry(CatalogReader.CatalogPropertyInterface):
    """Catalog property for retrieving the photometry at NNPZ format"""

    def _replace_nan(self, data):
        """
        Replace all the NaN flags
        """
        for flag in self.__nan_flags:
            data[data == flag] = np.nan

    def __init__(self, column_list, nan_flags=None):
        """Creates a new instance for the given column names.

        Args:
            column_list: A list of tuples with the columns containing the
                photometry. The first element of the tuple is the name of the
                value column and the second the error column.
                An optional third value is used as a multiplicative factor, and can be used
                to apply flux corrections.
            nan_flags: A list of values which when found in the photometry
                columns are replaced with NaN

        Note: The error columns are allowed to be None, in which case the errors
        of the band will be set to 0.
        """
        if len(column_list[0]) == 3:
            self.__column_list = column_list
        elif len(column_list[0]) == 2:
            self.__column_list = list(map(lambda t: t + (None,), column_list))
        else:
            self.__column_list = list(map(lambda t: (t,) + (None, None,), column_list))
        self.__nan_flags = nan_flags if nan_flags else []

    def __call__(self, catalog):
        """Returns the photometry of the catalog.

        Args:
            catalog: The catalog to read the photometry from

        Returns:
            A three dimensional numpy array of single precision floats where the
            first dimension has the same size as the number of objects the table
            contains photometries for, the second axis has same size as the
            given bands and the third axis has always size two, where the first
            element represents the photometry value and the second the
            uncertainty.

        Raises:
            UnknownNameException: If the given file does misses any of the
            expected columns
        """
        for value, error, correction in self.__column_list:
            if value not in catalog.colnames:
                raise UnknownNameException('Missing column {}'.format(value))
            if error is not None and error not in catalog.colnames:
                raise UnknownNameException('Missing column {}'.format(error))
            if correction is not None and correction not in catalog.colnames:
                raise UnknownNameException('Missing column {}'.format(correction))

        # Construct the result array with zeros
        data = np.zeros((len(catalog), len(self.__column_list), 2), dtype=np.float32)

        # Populate the data
        for i, (value, error, correction) in enumerate(self.__column_list):
            data[:, i, 0] = catalog[value]
            self._replace_nan(data[:, i, 0])
            if error is not None:
                data[:, i, 1] = catalog[error]
                self._replace_nan(data[:, i, 1])
            if correction is not None:
                data[:, i, 0] *= catalog[correction]

        return data
