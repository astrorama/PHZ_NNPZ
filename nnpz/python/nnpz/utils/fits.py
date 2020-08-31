#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
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

from __future__ import division, print_function

from astropy.io import fits


def npDtype2FitsTForm(data):
    """
        Generate a FITS format code appropriate for the type of this type of data.
    Args:
        data: numpy.array

    Returns: str

    """
    if data.dtype.kind in ('S', 'U'):
        fmt = "{}A".format(data.dtype.itemsize)
    else:
        dt = data.dtype.kind + str(data.dtype.alignment)
        fmt = fits.column.NUMPY2FITS[dt]
        if len(data.shape) > 1:
            fmt = "{}{}".format(data.shape[1], fmt)
    return fmt


def tableToHdu(table):
    """
    Convert an astropy Table object to a BinTableHDU.

    This method is a helper method for using astropy versions before 2.0. For
    newer versions of astropy the BinTableHDU constructor can be used directly.
    """
    columns = []
    for name in table.colnames:
        data = table[name].data
        fmt = npDtype2FitsTForm(data)
        columns.append(fits.Column(name=name, array=data, format=fmt))
    hdu = fits.BinTableHDU.from_columns(columns)
    for key, value in table.meta.items():
        if key == 'COMMENT':
            continue
        hdu.header.append((key, value))
    return hdu


def columnsToFitsColumn(columns):
    """
    Converts a list of astropy Table Columns to fits.Column
    """
    fits_cols = []
    for col in columns:
        data = col.data
        fmt = npDtype2FitsTForm(data)
        fits_cols.append(fits.Column(name=col.name, array=data, format=fmt))
    return fits_cols