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
import numpy as np
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
          domain there are potentially infinite possible values
    """

    def __init__(self, sampler: McSampler, param_name: str, binning: np.ndarray):
        self.__sampler = sampler
        self.__param_name = param_name
        self.__binning = binning
        self.__column = 'MC_COUNT_{}'.format(self.__param_name.upper())

    def get_column_definition(self):
        return [
            (self.__column, np.uint32, u.dimensionless_unscaled, len(self.__binning))
        ]

    def generate_output(self, indexes: np.ndarray, neighbor_info: np.ndarray,
                        output: np.ndarray):
        samples = self.__sampler.get_samples()[self.__param_name]
        output_col = output[self.__column]

        # Compute the binning as [i-0.5, i+0.5)
        bins = np.append(self.__binning, self.__binning[-1] + 1).astype(np.float32)
        bins -= 0.5

        # For each object, take a random weighted sample and generate the histogram
        for i in range(len(output)):
            output_col[i] = np.histogram(samples[i], bins=bins, density=False)[0]
