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
Created on: 20/04/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from astropy.table import Column
import numpy as np

from nnpz.io import OutputHandler
from nnpz import NnpzFlag


class Flags(OutputHandler.OutputColumnProviderInterface):

    def __init__(self, flag_list, separate_columns=False):
        self.__flag_list = flag_list
        self.__separate_columns = separate_columns

    def addContribution(self, reference_sample_i, neighbor, flags):
        pass

    def _separateColumns(self):
        columns = []
        for name in NnpzFlag.getFlagNames():
            columns.append(Column(np.asarray([f.isSet(NnpzFlag(name)) for f in self.__flag_list], dtype=np.bool), name))
        return columns

    def _byteColumns(self):
        flag_list_as_arrays = [f.asArray() for f in self.__flag_list]
        columns = []
        for i in range(NnpzFlag.getArraySize()):
            columns.append(Column(np.asarray([f[i] for f in flag_list_as_arrays], dtype=np.uint8), 'FLAGS_{}'.format(i+1)))
        return columns

    def getColumns(self):
        if self.__separate_columns:
            return self._separateColumns()
        else:
            return self._byteColumns()


