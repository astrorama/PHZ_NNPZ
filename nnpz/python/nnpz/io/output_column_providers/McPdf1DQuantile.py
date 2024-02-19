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


class McPdf1DQuantile(OutputHandler.OutputColumnProviderInterface):
    """
    Compute the quantile for a given parameter using a weighted random sample
    from the reference objects nearest to a given object

    See Also:
        nnpz.io.output_column_providers.McSampler
    Args:
        sampler: McSampler
            The handler that takes care of handling the weighted random sampling
        param_name:
            The parameter to compute the quantile for
        range_pcent:
            The range (in percent) for the quantile
    """

    def __init__(self, sampler: McSampler, param_name: str, range_pcent:float):
        super(McPdf1DQuantile, self).__init__()
        self.__sampler = sampler
        self.__param_name = param_name
        self.__range = range_pcent/100.0
        self.__column = 'PHZ_PP_{}_{}'.format(range_pcent, self.__param_name.upper())

    def get_column_definition(self):
        return [
            (self.__column, np.float32, u.dimensionless_unscaled, 2)
        ]

    def generate_output(self, indexes: np.ndarray, neighbor_info: np.ndarray,
                        output: np.ndarray):
        samples = self.__sampler.get_samples()
        param_samples = samples[self.__param_name]
        output_col = output[self.__column]
        for i in range(len(output)):
            output_col[i] = [np.quantile(param_samples[i],0.5-self.__range/2.0),np.quantile(param_samples[i],0.5+self.__range/2.0)]
