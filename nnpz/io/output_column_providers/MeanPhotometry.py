"""
Created on: 22/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np
from astropy.table import Column

from nnpz.exceptions import *
from nnpz.io import OutputHandler


class MeanPhotometry(OutputHandler.OutputColumnProviderInterface):


    def __init__(self, catalog_size, filter_names, data):

        if len(filter_names) != data.shape[1]:
            raise InvalidDimensionsException('Number of filter names does not match the data')

        self.__filter_names = filter_names
        self.__data = data

        self.__total_weights = np.zeros(catalog_size, dtype=np.float64)
        self.__total_values = np.zeros((catalog_size, len(filter_names)), dtype=np.float64)
        self.__total_errors = np.zeros((catalog_size, len(filter_names)), dtype=np.float64)


    def addContribution(self, reference_sample_i, catalog_i, weight, flags):

        phot = self.__data[reference_sample_i]
        self.__total_weights[catalog_i] += weight
        self.__total_values[catalog_i] += weight * phot[:,0]
        self.__total_errors[catalog_i] += weight * phot[:,1] * weight * phot[:,1]


    def getColumns(self):
        values = (self.__total_values.T / self.__total_weights).T
        errors = (np.sqrt(self.__total_errors).T / self.__total_weights).T
        columns = []
        for i, name in enumerate(self.__filter_names):
            columns.append(Column(values[:,i], 'mean_' + name, dtype=np.float32))
            columns.append(Column(errors[:,i], 'mean_' + name + '_ERR', dtype=np.float32))
        return columns
