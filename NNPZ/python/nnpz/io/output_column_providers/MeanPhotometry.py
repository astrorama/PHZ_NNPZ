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
    """
    Generate a list of columns with the mean photometry from the matching reference objects.
    The photometry coming from the reference objects are assumed to be on the reference color space
    (no reddening in the case of a reference sample).
    Reddening is optionally applied (photometry moved to the target color space) *after* the mean photometry
    is computed entirely on the reference color space.
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

        self.__filter_names = filter_names
        self.__data = data
        self.__unreddener = unreddener
        self.__target_ebv = target_ebv

        self.__total_weights = np.zeros(catalog_size, dtype=np.float64)
        self.__total_values = np.zeros((catalog_size, len(filter_names)), dtype=np.float64)
        self.__total_errors = np.zeros((catalog_size, len(filter_names)), dtype=np.float64)

    def addContribution(self, reference_sample_i, neighbor, flags):
        phot = self.__data[reference_sample_i]
        self.__total_weights[neighbor.index] += neighbor.weight
        self.__total_values[neighbor.index] += neighbor.weight * neighbor.scale * phot[:, 0]
        self.__total_errors[neighbor.index] += (neighbor.weight * neighbor.scale * phot[:, 1]) ** 2

    def getColumns(self):
        values = (self.__total_values.T / self.__total_weights).T
        errors = (np.sqrt(self.__total_errors).T / self.__total_weights).T

        if self.__unreddener:
            reddened = self.__unreddener.redden_data(np.stack([values, errors], axis=2), self.__target_ebv)
            values[:, :] = reddened[:, :, 0]
            errors[:, :] = reddened[:, :, 1]

        columns = []
        for i, name in enumerate(self.__filter_names):
            columns.append(Column(values[:, i], name + "_MEAN", dtype=np.float32))
            columns.append(Column(errors[:, i], name + '_MEAN_ERR', dtype=np.float32))
        return columns
