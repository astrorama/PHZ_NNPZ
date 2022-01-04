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
Created on: 01/02/18
Author: Nikolaos Apostolakos
"""


import numpy as np
from nnpz.io import OutputHandler


class MeanTrueRedshift(OutputHandler.OutputColumnProviderInterface):
    """
    Output a column with the redshift weighted mean. Note that this
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
        self.__total_weights = np.zeros(catalog_size, dtype=np.float64)
        self.__output = None

    def getColumnDefinition(self):
        return [
            ('REDSHIFT_MEAN', np.float32)
        ]

    def setWriteableArea(self, output_area):
        self.__output = output_area['REDSHIFT_MEAN']

    def addContribution(self, reference_sample_i, neighbor, flags):
        redshift = self.__ref_z[reference_sample_i]
        self.__total_weights[neighbor.index] += neighbor.weight
        self.__output[neighbor.index] += redshift * neighbor.weight

    def fillColumns(self):
        self.__output /= self.__total_weights
