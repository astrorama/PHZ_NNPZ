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
import numpy as np
from astropy.table import Table
from nnpz.io import OutputHandler


class McSliceAggregateBins(OutputHandler.OutputExtensionTableProviderInterface):
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

    def addContribution(self, reference_sample_i, neighbor, flags):
        pass

    def getExtensionTables(self):
        return {
            'BINS_MC_SLICE_AGGREGATE_{}_{}_{}'.format(
                self.__target_param.upper(), self.__slice_param.upper(), self.__suffix
            ): Table(
                {
                    self.__slice_param.upper(): self.__slice_binning.ravel(),
                })
        }
