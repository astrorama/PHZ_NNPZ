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
from astropy.table import Table
from nnpz.io import OutputHandler
from nnpz.io.output_column_providers.McCounter import McCounter


class McCounterBins(OutputHandler.OutputExtensionTableProviderInterface):
    """
    See McCounter
    """

    def __init__(self, counter: McCounter, param_name: str):
        self.__counter = counter
        self.__param_name = param_name

    def getExtensionTables(self):
        return {
            'BINS_MC_COUNT_{}'.format(self.__param_name.upper()): Table({
                'BINS': self.__counter.getBins(),
            })
        }
