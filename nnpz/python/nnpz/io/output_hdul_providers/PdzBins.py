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
Created on: 19/04/2018
Author: Alejandro Alvarez Ayllon
"""
import fitsio

from nnpz.io.OutputHandler import OutputHandler
from nnpz.reference_sample.ReferenceSample import ReferenceSample


class PdzBins(OutputHandler.OutputExtensionProviderInterface):
    """
    Generates an HDUL with the PDZ bins
    """

    def __init__(self, ref_sample: ReferenceSample):
        self.__bins = ref_sample.get_provider('pdz').get_redshift_bins()

    def add_extensions(self, fits: fitsio.FITS):
        fits.create_table_hdu({'BINS_PDF': self.__bins}, extname='BINS_PDF')
        fits['BINS_PDF'].write_column('BINS_PDF', self.__bins)
