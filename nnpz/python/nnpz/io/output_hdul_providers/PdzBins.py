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
Created on: 19/04/2018
Author: Alejandro Alvarez Ayllon
"""
from astropy.table import Table

from nnpz.io.OutputHandler import OutputHandler


class PdzBins(OutputHandler.OutputExtensionTableProviderInterface):
    """
    Generates an HDUL with the PDZ bins
    """

    def __init__(self, pdz_provider):
        self.__pdz_provider = pdz_provider

    def addContribution(self, reference_sample_i, neighbor, flags):
        pass

    def getExtensionTables(self):
        bins = self.__pdz_provider.getPdzBins()
        return {
            'BINS_PDF': Table({
                'BINS_PDF': bins,
            })
        }
