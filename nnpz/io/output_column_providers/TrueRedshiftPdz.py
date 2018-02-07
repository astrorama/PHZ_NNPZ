"""
Created on: 02/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np
from astropy.table import Column

from nnpz.io import OutputHandler


class TrueRedshiftPdz(OutputHandler.OutputColumnProviderInterface):

    def __init__(self, catalog_size, ref_true_redshift_list, z_min, z_max, bins_no):
        self.__ref_z = ref_true_redshift_list
        self.__z_min = z_min
        self.__z_max = z_max
        self.__pdz_bins = np.linspace(z_min, z_max, bins_no, dtype=np.float32)
        self.__step = self.__pdz_bins[1] - self.__pdz_bins[0]
        self.__pdzs = np.zeros((catalog_size, len(self.__pdz_bins)), dtype=np.float32)


    def addContribution(self, reference_sample_i, catalog_i, weight):
        z = self.__ref_z[reference_sample_i]
        pdz = self.__pdzs[catalog_i]

        pdz_i = int((z - self.__z_min + self.__step / 2.) / self.__step)
        if pdz_i < 0:
            pdz_i = 0
        if pdz_i >= len(self.__pdz_bins):
            pdz_i = self.__pdz_bins -1

        pdz[pdz_i] += weight


    def getColumns(self):
        integrals = np.trapz(self.__pdzs, axis=1)
        normalized = (self.__pdzs.T * integrals).T
        col = Column(normalized, 'TrueRedshiftPDZ')
        return [col]


    def getPdzBins(self):
        return self.__pdz_bins
