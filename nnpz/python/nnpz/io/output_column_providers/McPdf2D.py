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

import numpy as np
from nnpz.io import OutputHandler
from nnpz.io.output_column_providers.McSampler import McSampler


class McPdf2D(OutputHandler.OutputColumnProviderInterface):
    """
    Generate a two dimensional PDF for two given parameters using a weighted random sample
    from the reference objects nearest to a given object

    See Also:
        nnpz.io.output_hdul_providers.McPdf2DBins
        nnpz.io.output_column_providers.McSampler
    Args:
        sampler: McSampler
            The handler that takes care of handling the weighted random sampling
        param_names:
            A tuple with the two parameters to generate the 2D PDF for
        binning:
            A tuple with the two dimensional numpy array for the histogram binning
    """

    def __init__(self, sampler: McSampler,
                 param_names: Tuple[str, str], binning: Tuple[np.ndarray, np.ndarray]):
        super(McPdf2D, self).__init__()
        self.__sampler = sampler
        self.__param_names = list(param_names)
        self.__binning = binning
        self.__column = 'MC_PDF_2D_{}_{}'.format(*map(str.upper, self.__param_names))
        self.__output = None

    def getColumnDefinition(self):
        shape = (self.__binning[0].shape[0] - 1, self.__binning[1].shape[0] - 1)
        return [
            (self.__column, np.float32, shape)
        ]

    def setWriteableArea(self, output_area):
        self.__output = output_area[self.__column]
        print(self.__output.shape)

    def addContribution(self, reference_sample_i, neighbor, flags):
        """
        Does nothing for this provider, as the sampling is done by the McSampler
        """
        pass

    def fillColumns(self):
        """
        See OutputColumnProviderInterface.fillColumns
        """
        samples = self.__sampler.getSamples()

        for i, sample in enumerate(samples):
            self.__output[i, :] = np.histogram2d(
                sample[self.__param_names[0]], sample[self.__param_names[1]],
                bins=self.__binning, density=True
            )[0]
