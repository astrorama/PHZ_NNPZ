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
import astropy.units as u
import fitsio
import numpy as np
from nnpz.io import OutputHandler


class McPdf1DBins(OutputHandler.OutputExtensionProviderInterface):
    """
    Generate a an HDU with the binning associated to a 1D PDF
    See Also:
        nnpz.io.output_column_providers.McPdf1D
    Args:
        param_name:
            The parameter to generate the 1D PDF for
        binning:
            A one dimensional numpy array with the histogram binning
        unit:
            The unit for the parameter
    """

    def __init__(self, param_name: str, binning: np.ndarray, unit: u.Unit):
        self.__param_name = param_name
        # Take the bin center
        self.__binning = (binning[:-1] + binning[1:]) / 2.
        self.__unit = [str(unit)] if unit else None

    def add_extensions(self, fits: fitsio.FITS):
        extname = 'BINS_MC_PDF_1D_{}'.format(self.__param_name.upper())
        fits.create_table_hdu([self.__binning], names=['BINS_PDF'], units=self.__unit,
                              extname=extname)
        fits[extname].write_column('BINS_PDF', self.__binning)
