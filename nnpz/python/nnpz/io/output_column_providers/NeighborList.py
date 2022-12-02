#
# Copyright (C) 2012-2022 Euclid Science Ground Segment
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

from typing import List, Optional, Sequence, Tuple

import astropy.units as u
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

    def get_column_definition(self) \
        -> List[Tuple[str, np.dtype, u.Unit, Optional[Tuple[int, ...]]]]:
        return [
            (NEIGHBOR_IDS_COLNAME, np.int64, u.dimensionless_unscaled, self.__n_neighbors),
            (NEIGHBOR_WEIGHTS_COLNAME, np.float32, u.dimensionless_unscaled, self.__n_neighbors),
            (NEIGHBOR_SCALES_COLNAME, np.float32, u.dimensionless_unscaled, self.__n_neighbors)
        ]

    def generate_output(self, indexes: np.ndarray, neighbor_info: np.ndarray, output: np.ndarray):
        output[NEIGHBOR_IDS_COLNAME] = self.__ref_ids[neighbor_info['NEIGHBOR_INDEX']]
        output[NEIGHBOR_WEIGHTS_COLNAME] = neighbor_info['NEIGHBOR_WEIGHTS']
        output[NEIGHBOR_SCALES_COLNAME] = neighbor_info['NEIGHBOR_SCALING']
