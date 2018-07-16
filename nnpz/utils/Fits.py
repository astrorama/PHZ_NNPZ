"""
Created on: 22/02/2018
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from astropy.io import fits


def tableToHdu(table):
    """Converts an astropy Table object to a BinTableHDU.

    This method is a helper method for using astropy versions before 2.0. For
    newer versions of astropy the BinTableHDU constructor can be used directly.
    """
    columns = []
    for name in table.colnames:
        data = table[name].data
        dt = data.dtype.kind + str(data.dtype.alignment)
        format = fits.column.NUMPY2FITS[dt]
        if len(data.shape) > 1:
            format = "{}{}".format(data.shape[1], format)
        columns.append(fits.Column(name=name, array=data, format=format))
    hdu = fits.BinTableHDU.from_columns(columns)
    for key, value in table.meta.items():
        if key == 'COMMENT':
            continue
        hdu.header.append((key, value))
    return hdu


def columnsToFitsColumn(columns):
    fits_cols = []
    for col in columns:
        data = col.data
        dt = data.dtype.kind + str(data.dtype.alignment)
        format = fits.column.NUMPY2FITS[dt]
        if len(data.shape) > 1:
            format = "{}{}".format(data.shape[1], format)
        fits_cols.append(fits.Column(name=col.name, array=data, format=format))
    return fits_cols