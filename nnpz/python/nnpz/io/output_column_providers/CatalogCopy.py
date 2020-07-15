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
Created on: 01/03/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from nnpz.io import OutputHandler


class CatalogCopy(OutputHandler.OutputColumnProviderInterface):


    def __init__(self, catalog, columns = None):
        self.__catalog = catalog
        if columns is not None:
            self.__columns = columns
        else:
            self.__columns = self.__catalog.colnames


    def addContribution(self, reference_sample_i, neighbor, flags):
        pass


    def getColumns(self):
        return [c for _,c in self.__catalog[self.__columns].columns.items()]
