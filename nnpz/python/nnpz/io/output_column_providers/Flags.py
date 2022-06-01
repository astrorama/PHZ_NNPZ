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
Created on: 20/04/18
Author: Nikolaos Apostolakos
"""
from typing import List, Optional, Tuple

import astropy.units as u
import numpy as np
from nnpz.flags import NnpzFlag
from nnpz.io import OutputHandler


class Flags(OutputHandler.OutputColumnProviderInterface):
    """
    Generate the output column(s) with the flag values.

    Args:
        flag_list: list of NnpzFlag
        separate_columns: bool
            If True, each flag will be stored into an individual boolean column.
            Otherwise, they will be merged into a single integer column where each bit
            maps to a flag.
    """

    def __init__(self, separate_columns=False):
        self.__separate_columns = separate_columns
        self.__output_area = None

    def get_column_definition(self) \
            -> List[Tuple[str, np.dtype, u.Unit, Optional[Tuple[int, ...]]]]:
        if self.__separate_columns:
            return [
                (name, np.bool, u.dimensionless_unscaled) for name in NnpzFlag.FLAG_NAMES
            ]
        return [
            ('FLAGS', np.uint32, u.dimensionless_unscaled)
        ]

    def _separate_columns(self, flags: np.ndarray, output: np.ndarray):
        for flag, name in zip(NnpzFlag.FLAGS, NnpzFlag.FLAG_NAMES):
            output[name] = flags ^ flag

    def generate_output(self, indexes: np.ndarray, neighbor_info: np.ndarray, output: np.ndarray):
        if self.__separate_columns:
            self._separate_columns(neighbor_info['FLAGS'], output)
        else:
            np.copyto(output['FLAGS'], neighbor_info['FLAGS'], casting='same_kind')
