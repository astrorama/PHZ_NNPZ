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
from astropy.table import Column
from nnpz.io import OutputHandler
from nnpz.io.output_column_providers.McSampler import McSampler


class McCounter(OutputHandler.OutputColumnProviderInterface):
    """
    Similar to McPdf1D but for integer samples: store the counts themselves.
    In principle, McPdf1D could be used for this purpose, just making sure that the binning
    is adequately spaced between [i-0.5, i+0.5), for i an integer on the valid range of values.

    This output provider is for convenience so the configuration and output are more
    self-describing

    Note: By itself, it could count floating point as well, but the configuration will
          enforce only integral types are counted, as on any given range on the real
          domain there are potentially infinite posible values
    """

    def __init__(self, sampler: McSampler, param_name: str, binning: np.ndarray):
        self.__sampler = sampler
        self.__param_name = param_name
        self.__binning = binning

    def addContribution(self, reference_sample_i, neighbor, flags):
        """
        Does nothing for this provider, as the sampling is done by the McSampler
        """
        pass

    def getColumns(self):
        """
        See OutputColumnProviderInterface.getColumns
        """
        samples = self.__sampler.getSamples()[self.__param_name]

        # Compute the binning as [i-0.5, i+0.5)
        bins = np.append(self.__binning, self.__binning[-1] + 1).astype(np.float)
        bins -= 0.5

        # For each object, take a random weighted sample and generate the histogram
        counts = np.zeros((len(samples), bins.shape[0] - 1), dtype=np.int)

        for i in range(counts.shape[0]):
            counts[i, :] = np.histogram(samples[i], bins=bins, density=False)[0]

        return [
            Column(counts, 'MC_COUNT_{}'.format(self.__param_name.upper()))
        ]
