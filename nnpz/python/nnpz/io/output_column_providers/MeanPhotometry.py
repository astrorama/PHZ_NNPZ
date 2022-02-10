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
Created on: 22/02/18
Author: Nikolaos Apostolakos
"""
from typing import List, Optional, Tuple

import astropy.units as u
import numpy as np
from nnpz.io import OutputHandler
from nnpz.photometry.projection.source_independent_ebv import SourceIndependentGalacticEBV


class MeanPhotometry(OutputHandler.OutputColumnProviderInterface):
    """
    Generate a list of columns with the mean photometry from the matching reference objects.
    The photometry coming from the reference objects are assumed to be on the reference
    color space (no reddening in the case of a reference sample).
    Reddening is optionally applied (photometry moved to the target color space)
    *after* the mean photometry is computed entirely on the reference color space.
    """

    def __init__(self, filter_names: List[str], filter_idxs: np.ndarray, unit: u.Unit,
                 reddener: Optional[SourceIndependentGalacticEBV],
                 target_ebv: Optional[str]):
        """
        Constructor
        Args:
            filter_names:
                Names of the filters to be output
            filter_idxs:
                Indexes of the filters on the reference sample
            reddener: (Optional)
                An object implementing the method redden_data(photometry, ebv)
            target_ebv: (Optional)
                Target catalog extinction values
        """
        self.__filter_idxs = filter_idxs
        self.__unit = unit
        self.__unreddener = reddener
        self.__target_ebv = target_ebv

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
        for name, err, idx in zip(self.__columns, self.__err_columns, self.__filter_idxs):
            output[name] = np.average(ref_photo[:, :, idx, 0], weights=ref_weights, axis=-1)
            output[err] = np.average(ref_photo[:, :, idx, 1], weights=ref_weights, axis=-1)
