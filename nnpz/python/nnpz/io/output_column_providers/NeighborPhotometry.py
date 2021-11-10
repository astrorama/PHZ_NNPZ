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
from typing import List

import numpy as np
from nnpz.io import OutputHandler

NEIGHBOR_PHOTO_COLNAME = 'NEIGHBOR_PHOTOMETRY_{}'


class NeighborPhotometry(OutputHandler.OutputColumnProviderInterface):
    """
    Generate one column per filter, with the *matched* photometry values
    per neighbor.

    Args:
            filters: List of filters
            n_neighbors: Number of neighbors
    """

    def __init__(self, filters: List[str], n_neighbors: int):
        super(NeighborPhotometry, self).__init__()
        self.__filters = filters
        self.__n_neighbors = n_neighbors
        self.__n_photo = None

    def getColumnDefinition(self):
        cols = []
        for f in self.__filters:
            cols.append((
                NEIGHBOR_PHOTO_COLNAME.format(f), np.float32, self.__n_neighbors
            ))
        return cols

    def setWriteableArea(self, output_area):
        self.__n_photo = output_area

    def addContribution(self, reference_sample_i, neighbor, flags):
        for f in self.__filters:
            cname = NEIGHBOR_PHOTO_COLNAME.format(f)
            self.__n_photo[cname][neighbor.index, neighbor.position] = neighbor.matched_photo[f][0]

    def fillColumns(self):
        pass
