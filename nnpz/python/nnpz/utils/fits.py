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
Created on: 22/02/2018
Author: Nikolaos Apostolakos
"""


import numpy as np
from astropy.io import fits


def npDtype2FitsTForm(dtype, shape):
    """
    Generate a FITS format code appropriate for the type of this type of data.
    Args:
        dtype: type or numpy.dtype
        shape: a tuple or an integer
    Returns: str
    """
    if isinstance(dtype, type):
        dtype = np.dtype(dtype)
    if not hasattr(shape, '__len__'):
        shape = (shape,)
    if dtype.kind in ('S', 'U'):
        fmt = "{}A".format(dtype.itemsize)
    else:
        dt = dtype.kind + str(dtype.alignment)
        fmt = fits.column.NUMPY2FITS[dt]
        if len(shape) > 1:
            fmt = "{}{}".format(np.prod(shape[1:]), fmt)
    return fmt


def shape2FitsTDim(shape):
    """
    Generate a  FITS TDIM value if shape is multidimensional
    Args:
        shape: Shape. The first axis is assumed to be the number of rows
    """
    if len(shape) <= 2:
        return None
    return '(' + ','.join(map(str, reversed(shape[1:]))) + ')'


def tableToHdu(table):
    """
    Convert an astropy Table object to a BinTableHDU.

    This method is a helper method for using astropy versions before 2.0. For
    newer versions of astropy the BinTableHDU constructor can be used directly.
    """
    columns = []
    for name in table.colnames:
        data = table[name].data
        fmt = npDtype2FitsTForm(data.dtype, data.shape)
        columns.append(fits.Column(name=name, array=data, format=fmt))
    hdu = fits.BinTableHDU.from_columns(columns)
    for key, value in table.meta.items():
        if key == 'COMMENT':
            continue
        hdu.header.append((key, value))
    return hdu
