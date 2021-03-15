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
Created on: 02/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from typing import Sequence

import numpy as np
from astropy.table import Column
from nnpz.io import OutputHandler

NEIGHBOR_IDS_COLNAME = 'NEIGHBOR_IDS'
NEIGHBOR_WEIGHTS_COLNAME = 'NEIGHBOR_WEIGHTS'
NEIGHBOR_SCALES_COLNAME = 'NEIGHBOR_SCALING'


class NeighborList(OutputHandler.OutputColumnProviderInterface):
    """
    Generate three columns with information about the neighbors: ids, weights and the
    applied scaling
    """

    def __init__(self, catalog_size: int, ref_ids: Sequence, n_neighbors: int):
        self.__neighbors = np.full((catalog_size, n_neighbors), fill_value=-1, dtype=int)
        self.__weights = np.full((catalog_size, n_neighbors), fill_value=np.nan, dtype=np.float32)
        self.__scales = np.full((catalog_size, n_neighbors), fill_value=np.nan, dtype=np.float32)
        self.__ref_ids = ref_ids

    def addContribution(self, reference_sample_i, neighbor, flags):
        self.__neighbors[neighbor.index, neighbor.position] = self.__ref_ids[reference_sample_i]
        self.__weights[neighbor.index, neighbor.position] = neighbor.weight
        self.__scales[neighbor.index, neighbor.position] = neighbor.scale

    def getColumns(self):
        neighbor_col = Column(self.__neighbors, NEIGHBOR_IDS_COLNAME)
        weight_col = Column(self.__weights, NEIGHBOR_WEIGHTS_COLNAME)
        scale_col = Column(self.__scales, NEIGHBOR_SCALES_COLNAME)
        return [neighbor_col, weight_col, scale_col]
