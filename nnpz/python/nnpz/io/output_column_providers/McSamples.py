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

from astropy.table import Column
from nnpz.io import OutputHandler
from nnpz.io.output_column_providers.McSampler import McSampler


class McSamples(OutputHandler.OutputColumnProviderInterface):
    """
    Store the samples themselves into the output catalog
    """

    def __init__(self, sampler: McSampler, parameters: Tuple[str]):
        self.__sampler = sampler
        self.__params = parameters

    def addContribution(self, reference_sample_i, neighbor, flags):
        """
        Does nothing for this provider, as the sampling is done by the McSampler
        """
        pass

    def getColumns(self):
        """
        See OutputColumnProviderInterface.getColumns
        """
        samples = self.__sampler.getSamples()
        if self.__params is None:
            self.__params = samples.dtype.names

        columns = []
        for param in self.__params:
            columns.append(Column(
                data=samples[param], name='MC_SAMPLES_' + param.upper()
            ))
        return columns
