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
from typing import Tuple

import numpy as np
from astropy.table import Column
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
        pdfs = np.zeros((len(samples),
                         self.__binning[0].shape[0] - 1,
                         self.__binning[1].shape[0] - 1), dtype=np.float)

        for i in range(len(samples)):
            pdfs[i, :] = np.histogram2d(
                samples[i][self.__param_names[0]], samples[i][self.__param_names[1]],
                bins=self.__binning, density=True
            )[0]

        return [
            Column(
                data=pdfs.reshape(len(samples), -1),
                name='MC_PDF_2D_{}_{}'.format(*map(str.upper, self.__param_names)),
            )
        ]
