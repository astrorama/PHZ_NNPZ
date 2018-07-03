"""
Created on: 15/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np
from astropy.table import Column

from nnpz.io import OutputHandler


class MedianTrueRedshift(OutputHandler.OutputColumnProviderInterface):

    def __init__(self, catalog_size, ref_true_redshift_list):
        self.__ref_z = ref_true_redshift_list
        self.__zs = [[] for i in range(catalog_size)]
        self.__weights = [[] for i in range(catalog_size)]


    def addContribution(self, reference_sample_i, catalog_i, weight, flags):
        z = self.__ref_z[reference_sample_i]
        self.__zs[catalog_i].append(z)
        self.__weights[catalog_i].append(weight)


    def getColumns(self):
        median_z = np.zeros(len(self.__zs), dtype=np.float32)
        for i, (z, w) in enumerate(zip(self.__zs, self.__weights)):
            half = sum(w) / 2.
            c = 0
            for sort_i in np.argsort(z):
                c += w[sort_i]
                if c > half:
                    median_z[i] = z[sort_i]
                    break
        col = Column(median_z, 'REDSHIFT_MEDIAN')
        return [col]