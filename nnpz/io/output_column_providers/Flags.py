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

    def addContribution(self, reference_sample_i, catalog_i, weight, flags):
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


