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
Created on: 22/02/18
Author: Nikolaos Apostolakos
"""
from typing import List, Optional, Tuple

import astropy.units as u
import numpy as np
from nnpz.io import OutputHandler


class MeanPhotometry(OutputHandler.OutputColumnProviderInterface):
    """
    Generate a list of columns with the mean photometry from the matching reference objects.
    The photometry coming from the reference objects are assumed to be on the reference
    color space (no reddening in the case of a reference sample).
    Reddening is optionally applied (photometry moved to the target color space)
    *after* the mean photometry is computed entirely on the reference color space.
    """

    def __init__(self, filter_names: List[str], filter_idxs: np.ndarray, unit: u.Unit):
        """
        Constructor
        Args:
            filter_names:
                Names of the filters to be output
            filter_idxs:
                Indexes of the filters on the reference sample
        """
        self.__filter_idxs = filter_idxs
        self.__unit = unit
        self.__columns = [name + '_MEAN' for name in filter_names]
        self.__err_columns = [name + '_MEAN_ERR' for name in filter_names]

    def get_column_definition(self) \
        -> List[Tuple[str, np.dtype, u.Unit, Optional[Tuple[int, ...]]]]:
        col_defs = []
        for name, err in zip(self.__columns, self.__err_columns):
            col_defs.append((name, np.float32, self.__unit))
            col_defs.append((err, np.float32, self.__unit))
        return col_defs

    def generate_output(self, indexes: np.ndarray, neighbor_info: np.ndarray, output: np.ndarray):
        ref_photo = neighbor_info['NEIGHBOR_PHOTOMETRY']
        ref_weights = neighbor_info['NEIGHBOR_WEIGHTS']
        weights_sum = np.sum(ref_weights, axis=1)
        mask = weights_sum==0
        print("Warning : " + str(np.sum(mask)) + " objects have zero weights sum")
        ref_weights[mask,:]=1
        for name, err, idx in zip(self.__columns, self.__err_columns, self.__filter_idxs):
            output[name] = np.average(ref_photo[:, :, idx, 0], weights=ref_weights, axis=-1)
            output[err] = np.average(ref_photo[:, :, idx, 1], weights=ref_weights, axis=-1)
