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
Created on: 20/04/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np
from nnpz import NnpzFlag
from nnpz.io import OutputHandler


# FIXME: Probably better to have two different output handlers implementing
#        separate/joint flag columns

class Flags(OutputHandler.OutputColumnProviderInterface):
    """
    Generate the output column(s) with the flag values.

    Args:
        flag_list: list of NnpzFlag
        separate_columns: bool
            If True, each flag will be stored into an individual boolean column.
            Otherwise, they will be merged into a single integer column where each bit
            maps to a flag.
    """

    def __init__(self, flag_list, separate_columns=False):
        self.__flag_list = flag_list
        self.__separate_columns = separate_columns
        self.__output_area = None

    def getColumnDefinition(self):
        if self.__separate_columns:
            return [
                (name, np.bool) for name in NnpzFlag.getFlagNames()
            ]
        return [
            ('FLAGS_{}'.format(i + 1), np.uint8)
            for i in range(NnpzFlag.getArraySize())
        ]

    def setWriteableArea(self, output_area):
        self.__output_area = output_area

    def addContribution(self, reference_sample_i, neighbor, flags):
        pass

    def _separateColumns(self):
        for name in NnpzFlag.getFlagNames():
            self.__output_area[name] = [f.isSet(NnpzFlag(name)) for f in self.__flag_list]

    def _byteColumns(self):
        flag_list_as_arrays = [f.asArray() for f in self.__flag_list]
        for i in range(NnpzFlag.getArraySize()):
            name = 'FLAGS_{}'.format(i + 1)
            self.__output_area[name] = [flag[i] for flag in flag_list_as_arrays]

    def fillColumns(self):
        if self.__separate_columns:
            return self._separateColumns()
        return self._byteColumns()
