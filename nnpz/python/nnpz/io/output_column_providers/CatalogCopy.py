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
Created on: 01/03/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from nnpz.io import OutputHandler


class CatalogCopy(OutputHandler.OutputColumnProviderInterface):
    """
    Copy a list of columns from the target catalog to the output
    Args:
        catalog: astropy.table.Table
            Target catalog
        columns: list
            List of column names to copy
    """

    def __init__(self, catalog, columns=None):
        self.__catalog = catalog
        if columns is not None:
            self.__columns = columns
        else:
            self.__columns = self.__catalog.colnames
        self.__output_area = None

    def setWriteableArea(self, output_area):
        self.__output_area = output_area

    def getColumnDefinition(self):
        defs = []
        for c in self.__columns:
            defs.append((c, self.__catalog[c].dtype, self.__catalog[c].shape[1:]))
        return defs

    def addContribution(self, reference_sample_i, neighbor, flags):
        pass

    def fillColumns(self):
        for c in self.__columns:
            self.__output_area[c] = self.__catalog[c]
