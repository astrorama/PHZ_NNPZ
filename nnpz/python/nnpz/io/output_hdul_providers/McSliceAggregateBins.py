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
import numpy as np
from astrometry.util.fits import fitsio
from nnpz.io import OutputHandler


class McSliceAggregateBins(OutputHandler.OutputExtensionProviderInterface):
    """
    Store an extension table with the binning used for the sliced aggregate values

    See Also
        McSliceAggregate
    """

    def __init__(self, target_param: str, slice_param: str, suffix: str, slice_binning: np.ndarray):
        self.__target_param = target_param
        self.__slice_param = slice_param
        self.__suffix = suffix
        self.__slice_binning = slice_binning

    def add_extensions(self, fits: fitsio.FITS):
        target_col = self.__target_param.upper()
        slice_col = self.__slice_param.upper()
        extname = 'BINS_MC_SLICE_AGGREGATE_{}_{}_{}'.format(target_col, slice_col, self.__suffix)
        val = self.__slice_binning.ravel()
        fits.create_table_hdu({slice_col: val}, extname=extname)
        fits[extname].write_column(slice_col, val)
