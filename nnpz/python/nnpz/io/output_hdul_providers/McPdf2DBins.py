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
from typing import Tuple

import fitsio
import numpy as np
from nnpz.io import OutputHandler


class McPdf2DBins(OutputHandler.OutputExtensionProviderInterface):
    """
    Generate a an HDU with the binning associated to a 2D PDF
    See Also:
        nnpz.io.output_column_providers.McPdf2D
    Args:
        param_names:
            The two parameters to generate the 2D PDF for
        binning:
            A one dimensional numpy array with the histogram binning
    """

    def __init__(self, param_names: Tuple[str, str], binning: Tuple[np.ndarray, np.ndarray]):
        self.__param_names = param_names
        # Take the bin center
        bin1 = (binning[0][:-1] + binning[0][1:]) / 2.
        bin2 = (binning[1][:-1] + binning[1][1:]) / 2.
        self.__binning = np.meshgrid(bin1, bin2)

    def add_extensions(self, fits: fitsio.FITS):
        col1 = self.__param_names[0].upper()
        col2 = self.__param_names[1].upper()
        extname = 'BINS_MC_PDF_2D_{}_{}'.format(col1, col2)
        val1 = self.__binning[0].T.ravel()
        val2 = self.__binning[1].T.ravel()
        fits.create_table_hdu({col1: val1, col2: val2}, extname=extname)
        fits[extname].write([val1, val2], columns=[col1, col2])
