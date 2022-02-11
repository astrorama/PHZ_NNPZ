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
Created on: 01/03/18
Author: Nikolaos Apostolakos
"""
from typing import List, Optional, Tuple

import astropy.units as u
import fitsio.hdu
import numpy as np
from fitsio.hdu.table import _tdim2shape
from nnpz.io import OutputHandler


class CatalogCopy(OutputHandler.OutputColumnProviderInterface):
    """
    Copy a list of columns from the target catalog to the output
    Args:
        catalog: astropy.table.Table
            Target catalog
        columns: list
            List of column names to copy
    """

    def __get_details(self, header: fitsio.FITSHDR) \
            -> Tuple[List[u.Unit], List[Optional[Tuple[int, ...]]]]:
        units = [''] * len(self.__columns)
        shapes = [None] * len(self.__columns)
        i = 1
        col_name = header.get(f'TTYPE{i}')
        while col_name:
            unit_str = header.get(f'TUNIT{i}')
            dim_str = header.get(f'TDIM{i}')
            if col_name in self.__columns:
                col_idx = self.__columns.index(col_name)
                if unit_str:
                    units[col_idx] = u.Unit(unit_str)
                if dim_str:
                    shapes[col_idx] = _tdim2shape(dim_str, name=col_name)
            i += 1
            col_name = header.get(f'TTYPE{i}')
        return units, shapes

    def __init__(self, columns: np.dtype, catalog: fitsio.hdu.TableHDU):
        self.__columns = []
        self.__dtype = []
        for name in columns.names:
            self.__columns.append(name)
            self.__dtype.append(columns[name])
        self.__units, self.__shapes = self.__get_details(catalog.read_header())
        self.__catalog = catalog

    def get_column_definition(self) \
            -> List[Tuple[str, np.dtype, u.Unit, Optional[Tuple[int, ...]]]]:
        defs = []
        for col, dtype, unit, shape in zip(self.__columns, self.__dtype, self.__units,
                                           self.__shapes):
            if shape is not None:
                defs.append((col, dtype, unit, shape))
            else:
                defs.append((col, dtype, unit))
        return defs

    def generate_output(self, indexes: np.ndarray, neighbor_info: np.ndarray, output: np.ndarray):
        for c in self.__columns:
            output[c] = self.__catalog.read_column(c, rows=indexes)
