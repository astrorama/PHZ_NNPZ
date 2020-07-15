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

    def __init__(self, catalog_size, ref_true_redshift_list):
        self.__ref_z = ref_true_redshift_list
        self.__zs = [[] for i in range(catalog_size)]
        self.__weights = [[] for i in range(catalog_size)]


    def addContribution(self, reference_sample_i, neighbor, flags):
        z = self.__ref_z[reference_sample_i]
        self.__zs[neighbor.index].append(z)
        self.__weights[neighbor.index].append(neighbor.weight)


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
