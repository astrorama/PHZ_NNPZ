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
Created on: 15/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np
from astropy.table import Column

from nnpz.io import OutputHandler


class MedianTrueRedshift(OutputHandler.OutputColumnProviderInterface):
    """
    Output a column with the redshift weighted median. Note that this
    works only for reference catalogs with a point estimate for the redshift.
    For reference samples we have a PDF for the redshift.

    Args:
            catalog_size: int
                Number of target objects, for memory allocation
            ref_true_redshift_list: list or np.array of float
                True redshift for the reference objects.
    """

    def __init__(self, catalog_size, ref_true_redshift_list):
        self.__ref_z = ref_true_redshift_list
        self.__zs = [[] for i in range(catalog_size)]
        self.__weights = [[] for i in range(catalog_size)]

    def addContribution(self, reference_sample_i, neighbor, flags):
        redshift = self.__ref_z[reference_sample_i]
        self.__zs[neighbor.index].append(redshift)
        self.__weights[neighbor.index].append(neighbor.weight)

    def getColumns(self):
        median_z = np.zeros(len(self.__zs), dtype=np.float32)
        for i, (redshift, weight) in enumerate(zip(self.__zs, self.__weights)):
            half = sum(weight) / 2.
            c = 0
            for sort_i in np.argsort(redshift):
                c += weight[sort_i]
                if c > half:
                    median_z[i] = redshift[sort_i]
                    break
        col = Column(median_z, 'REDSHIFT_MEDIAN')
        return [col]