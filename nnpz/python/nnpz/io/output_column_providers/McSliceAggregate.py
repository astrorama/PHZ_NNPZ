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
from typing import Callable

import numpy as np
from astropy.table import Column
from nnpz.io import OutputHandler
from nnpz.io.output_column_providers.McSampler import McSampler


class McSliceAggregate(OutputHandler.OutputColumnProviderInterface):
    """
    This output provider takes two MC parameters, say A and B, a callable and
    a binning for B. It then uses the callable to compute an aggregate of A over
    slices of B.
    The callable must accept the 'axis' keyword (i.e. np.average)
    """

    def __init__(self, sampler: McSampler, target_param: str, slice_param: str,
                 suffix: str, aggregator: Callable, binning: np.ndarray):
        self.__sampler = sampler
        self.__target_param = target_param
        self.__slice_param = slice_param
        self.__suffix = suffix
        self.__aggregator = aggregator
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
        samples = self.__sampler.getSamples()

        aggregated = np.zeros((len(samples), len(self.__binning) - 1), dtype=np.float)

        for i in range(aggregated.shape[1]):
            rmin, rmax = self.__binning[i:i + 2]
            mask = (samples[self.__slice_param] < rmin) | (samples[self.__slice_param] >= rmax)
            data = np.ma.array(samples[self.__target_param], mask=mask, copy=False)
            # Technically NaN is more adequate for the mean of an empty set, but this breaks
            # boost on later stages of the PHZ pipeline :(
            aggregated[:, i] = self.__aggregator(data, axis=1).filled(-99.)

        return [
            Column(
                data=aggregated,
                name='MC_SLICE_AGGREGATE_{}_{}_{}'.format(
                    self.__target_param.upper(), self.__slice_param.upper(), self.__suffix
                )
            )
        ]
