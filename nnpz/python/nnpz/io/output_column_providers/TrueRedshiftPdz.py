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
Created on: 02/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np
from astropy.table import Column

from nnpz.io import OutputHandler


class TrueRedshiftPdz(OutputHandler.OutputColumnProviderInterface):
    """
    Generate a PDZ from redshift point estimates. Note that this
    works only for reference catalogs with a point estimate for the redshift.
    For catalogs with PDZ, or for reference samples, use CoaddedPdz instead.

    Args:
        catalog_size: int
            Number of target objects, for memory allocation
        ref_true_redshift_list: list or np.array of float
            True redshift for the reference objects.
        z_min: float
            Minimum redshift
        z_max:
            Maximum redshift
        bins_no:
            Number of bins
    """

    def __init__(self, catalog_size, ref_true_redshift_list, z_min, z_max, bins_no):
        self.__ref_z = ref_true_redshift_list
        self.__z_min = z_min
        self.__z_max = z_max
        self.__pdz_bins = np.linspace(z_min, z_max, bins_no, dtype=np.float64)
        self.__step = self.__pdz_bins[1] - self.__pdz_bins[0]
        self.__pdzs = np.zeros((catalog_size, len(self.__pdz_bins)), dtype=np.float64)

    def addContribution(self, reference_sample_i, neighbor, flags):
        redshift = self.__ref_z[reference_sample_i]
        pdz = self.__pdzs[neighbor.index]

        pdz_i = int((redshift - self.__z_min + self.__step / 2.) / self.__step)
        if pdz_i < 0:
            pdz_i = 0
        if pdz_i >= len(self.__pdz_bins):
            pdz_i = self.__pdz_bins - 1

        pdz[pdz_i] += neighbor.weight

    def getColumns(self):
        integrals = 1. / np.trapz(self.__pdzs, self.__pdz_bins, axis=1)
        normalized = (self.__pdzs.T * integrals).T
        col = Column(normalized, 'REDSHIFT_PDF', dtype=np.float32)
        return [col]

    def getPdzBins(self):
        return self.__pdz_bins
