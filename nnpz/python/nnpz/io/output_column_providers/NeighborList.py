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


from typing import Sequence

import numpy as np
from nnpz.io import OutputHandler

NEIGHBOR_IDS_COLNAME = 'NEIGHBOR_IDS'
NEIGHBOR_WEIGHTS_COLNAME = 'NEIGHBOR_WEIGHTS'
NEIGHBOR_SCALES_COLNAME = 'NEIGHBOR_SCALING'


class NeighborList(OutputHandler.OutputColumnProviderInterface):
    """
    Generate three columns with information about the neighbors: ids, weights and the
    applied scaling
    """

    def __init__(self, ref_ids: Sequence, n_neighbors: int):
        super(NeighborList, self).__init__()
        self.__ref_ids = ref_ids
        self.__n_neighbors = n_neighbors
        self.__neighbors = None
        self.__weights = None
        self.__scales = None

    def getColumnDefinition(self):
        return [
            (NEIGHBOR_IDS_COLNAME, np.int64, self.__n_neighbors),
            (NEIGHBOR_WEIGHTS_COLNAME, np.float32, self.__n_neighbors),
            (NEIGHBOR_SCALES_COLNAME, np.float32, self.__n_neighbors)
        ]

    def setWriteableArea(self, output_area):
        self.__neighbors = output_area[NEIGHBOR_IDS_COLNAME]
        self.__weights = output_area[NEIGHBOR_WEIGHTS_COLNAME]
        self.__scales = output_area[NEIGHBOR_SCALES_COLNAME]

    def addContribution(self, reference_sample_i, neighbor, flags):
        self.__neighbors[neighbor.index, neighbor.position] = self.__ref_ids[reference_sample_i]
        self.__weights[neighbor.index, neighbor.position] = neighbor.weight
        self.__scales[neighbor.index, neighbor.position] = neighbor.scale

    def fillColumns(self):
        pass
