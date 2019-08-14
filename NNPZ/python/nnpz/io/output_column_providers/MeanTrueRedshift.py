"""
Created on: 01/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np
from astropy.table import Column

from nnpz.io import OutputHandler


class MeanTrueRedshift(OutputHandler.OutputColumnProviderInterface):

    def __init__(self, catalog_size, ref_true_redshift_list):
        self.__ref_z = ref_true_redshift_list
        self.__total_weights = np.zeros(catalog_size, dtype=np.float64)
        self.__sums = np.zeros(catalog_size, dtype=np.float64)


    def addContribution(self, reference_sample_i, catalog_i, weight, flags):
        z = self.__ref_z[reference_sample_i]
        self.__total_weights[catalog_i] += weight
        self.__sums[catalog_i] += z * weight


    def getColumns(self):
        mean_z = self.__sums / self.__total_weights
        col = Column(mean_z, 'REDSHIFT_MEAN')
        return [col]