"""
Created on: 28/03/2018
Author: Alejandro Alvarez Ayllon
"""

from __future__ import division, print_function

import numpy as np
from astropy.table import Column
from nnpz.io import OutputHandler


class CoaddedPdz(OutputHandler.OutputColumnProviderInterface):
    """
    The CoaddedPdz output provider generates a PDZ out of the weighted sum
    of the neighbors PDZ, and re-normalized (its integral is 1)
    It generates two columns:
    - CoaddedPdz, with the PDZ values
    - CoaddedPdzBins, with the PDZ bins
    It assumes all reference samples have the same PDZ bins. If not, it will raise
    an assert error when a mismatch is found.
    """

    def __init__(self, catalog_size, reference_sample, ref_ids):
        super(CoaddedPdz, self).__init__()
        self.__reference_sample = reference_sample
        self.__pdz_bins = reference_sample.getPdzData(ref_ids[0])[:, 0]
        self.__pdzs = np.zeros((catalog_size, len(self.__pdz_bins)), dtype=np.float32)
        self.__ref_ids = ref_ids
        self.__current_ref_i = None
        self.__current_ref_pdz = None

    def addContribution(self, reference_sample_i, catalog_i, weight, flags):
        if reference_sample_i != self.__current_ref_i:
            ref_id = self.__ref_ids[reference_sample_i]
            self.__current_ref_i = reference_sample_i
            self.__current_ref_pdz = self.__reference_sample.getPdzData(ref_id)
            assert (self.__current_ref_pdz[:, 0] == self.__pdz_bins).all()

        self.__pdzs[catalog_i] += self.__current_ref_pdz[:, 1] * weight

    def getColumns(self):
        integrals = 1. / np.trapz(self.__pdzs, self.__pdz_bins, axis=1)
        normalized = (self.__pdzs.T * integrals).T
        return [
            Column(normalized, 'CoaddedPdz'),
            Column([self.__pdz_bins] * normalized.shape[0], 'CoaddedPdzBins'),
        ]

    def getPdzBins(self):
        return self.__pdz_bins
