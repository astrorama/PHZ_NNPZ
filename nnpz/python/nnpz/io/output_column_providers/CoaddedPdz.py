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
Created on: 28/03/2018
Author: Alejandro Alvarez Ayllon
"""

from __future__ import division, print_function

import numpy as np
from nnpz.io import OutputHandler

smallest_f32 = np.finfo(np.float32).tiny
biggest_f32 = np.finfo(np.float32).max


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
        self.__ref_ids = ref_ids
        self.__current_ref_i = None
        self.__current_ref_pdz = None
        self.__pdzs = None
        self.__scales = np.ones(catalog_size, dtype=np.float64)

    def getColumnDefinition(self):
        return [
            ('REDSHIFT_PDF', np.float32, len(self.__pdz_bins))
        ]

    def setWriteableArea(self, output_area):
        self.__pdzs = output_area['REDSHIFT_PDF']

    def addContribution(self, reference_sample_i, neighbor, flags):
        # If the weight is barely a floating point error, ignore
        if neighbor.weight <= 1e-300:
            return

        if reference_sample_i != self.__current_ref_i:
            ref_id = self.__ref_ids[reference_sample_i]
            self.__current_ref_i = reference_sample_i
            self.__current_ref_pdz = self.__reference_sample.getPdzData(ref_id)[:, 1].astype(
                np.float64)

        # Note that pdz is float64, so we use it as working area
        with np.errstate(over='ignore'):
            pdz = self.__current_ref_pdz * neighbor.weight * self.__scales[neighbor.index]
        pdz += self.__pdzs[neighbor.index]

        # Re-scale if necessary so it fits on 32 bits floating point
        if np.all(pdz <= smallest_f32) & np.any(pdz > 0):
            self.__scales[neighbor.index] = 1 / np.max(pdz)
            pdz *= self.__scales[neighbor.index]
        # If a previous neighbor had a weight order of magnitudes lower than this one,
        # we need to lower the scaling
        elif np.any(pdz >= biggest_f32):
            np.copyto(pdz, self.__pdzs[neighbor.index])
            pdz /= self.__scales[neighbor.index]
            pdz += self.__current_ref_pdz * neighbor.weight
            self.__scales[neighbor.index] = 1 / np.max(pdz)
            pdz *= self.__scales[neighbor.index]
        self.__pdzs[neighbor.index] = pdz

    def fillColumns(self):
        # Running trapz over all pdzs would be faster, but it can cause a big allocation
        for i in range(len(self.__pdzs)):
            integral = 1. / np.trapz(self.__pdzs[i], self.__pdz_bins)
            self.__pdzs[i] = self.__pdzs[i] * integral

    def getPdzBins(self):
        """
        Returns: np.array
            PDZ bins
        """
        return self.__pdz_bins

    def getPdz(self):
        return self.__pdzs
