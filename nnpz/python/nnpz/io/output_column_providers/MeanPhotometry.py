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
Created on: 22/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np
from nnpz.exceptions import InvalidDimensionsException
from nnpz.io import OutputHandler
from nnpz.utils.numpy import recarray_flat_view


class MeanPhotometry(OutputHandler.OutputColumnProviderInterface):
    """
    Generate a list of columns with the mean photometry from the matching reference objects.
    The photometry coming from the reference objects are assumed to be on the reference
    color space (no reddening in the case of a reference sample).
    Reddening is optionally applied (photometry moved to the target color space)
    *after* the mean photometry is computed entirely on the reference color space.
    """

    def __init__(self, catalog_size, filter_names, data, unreddener, target_ebv):
        """
        Constructor
        Args:
            catalog_size:
                Number of elements in the target catalog
            filter_names:
                Names of the filters to be output
            data:
                Reference sample photometry
            unreddener: (Optional)
                An object implementing the method redden_data(photometry, ebv)
            target_ebv: (Optional)
                Target catalog extinction values
        """

        if len(filter_names) != data.shape[1]:
            raise InvalidDimensionsException('Number of filter names does not match the data')

        self.__data = data
        self.__unreddener = unreddener
        self.__target_ebv = target_ebv

        self.__columns = [name + '_MEAN' for name in filter_names]
        self.__err_columns = [name + '_MEAN_ERR' for name in filter_names]

        self.__total_weights = np.zeros((catalog_size, 1), dtype=np.float64)
        self.__total_values = None
        self.__total_errors = None

    def getColumnDefinition(self):
        col_defs = []
        for name in self.__columns:
            col_defs.append((name, np.float32))
        for err_name in self.__err_columns:
            col_defs.append((err_name, np.float32))
        return col_defs

    def setWriteableArea(self, output_area):
        self.__total_values = recarray_flat_view(output_area, self.__columns)
        self.__total_errors = recarray_flat_view(output_area, self.__err_columns)

    def addContribution(self, reference_sample_i, neighbor, flags):
        ref_phot = self.__data[reference_sample_i]
        self.__total_weights[neighbor.index] += neighbor.weight
        self.__total_values[neighbor.index] += neighbor.weight * neighbor.scale * ref_phot[:, 0]
        self.__total_errors[neighbor.index] += (neighbor.weight * neighbor.scale * ref_phot[:,
                                                                                   1]) ** 2

    def fillColumns(self):
        self.__total_values /= self.__total_weights
        np.sqrt(self.__total_errors, out=self.__total_errors)
        self.__total_errors /= self.__total_weights

        if self.__unreddener:
            # FIXME: This can probably be done better, too many copies are involved
            photometry = np.stack([self.__total_values, self.__total_errors], axis=2)
            reddened = self.__unreddener.redden_data(photometry, self.__target_ebv)
            self.__total_values[:, :] = reddened[:, :, 0]
            self.__total_errors[:, :] = reddened[:, :, 1]
