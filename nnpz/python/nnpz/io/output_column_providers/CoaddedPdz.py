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
    It assumes all reference samples have the same PDZ bins.
    """

    def __init__(self, catalog_size, reference_sample, ref_ids):
        super(CoaddedPdz, self).__init__()
        self.__reference_sample = reference_sample
        self.__pdz_bins = reference_sample.getPdzData(ref_ids[0])[:, 0]
        self.__pdzs = np.zeros((catalog_size, len(self.__pdz_bins)), dtype=np.float64)
        self.__ref_ids = ref_ids
        self.__current_ref_i = None
        self.__current_ref_pdz = None

    def addContribution(self, reference_sample_i, neighbor, flags):
        if reference_sample_i != self.__current_ref_i:
            ref_id = self.__ref_ids[reference_sample_i]
            self.__current_ref_i = reference_sample_i
            self.__current_ref_pdz = np.array(self.__reference_sample.getPdzData(ref_id), dtype=np.float64)
            if not (self.__current_ref_pdz[:, 0] == self.__pdz_bins).all():
                raise ValueError('Invalid number of PDZ bins')

        self.__pdzs[neighbor.index] += self.__current_ref_pdz[:, 1] * neighbor.weight

    def getColumns(self):
        integrals = 1. / np.trapz(self.__pdzs, self.__pdz_bins, axis=1)
        normalized = (self.__pdzs.T * integrals).T
        return [
            Column(normalized, 'REDSHIFT_PDF', dtype=np.float32),
        ]

    def getPdzBins(self):
        """
        Returns: np.array
            PDZ bins
        """
        return self.__pdz_bins