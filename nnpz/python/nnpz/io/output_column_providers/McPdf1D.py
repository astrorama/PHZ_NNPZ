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

import numpy as np
from astropy.table import Column
from nnpz.io import OutputHandler
from nnpz.io.output_column_providers.McSampler import McSampler


class McPdf1D(OutputHandler.OutputColumnProviderInterface):
    """
    Generate a one dimensional PDF for a given parameter using a weighted random sample
    from the reference objects nearest to a given object

    See Also:
        nnpz.io.output_hdul_providers.McPdf1DBins
        nnpz.io.output_column_providers.McSampler
    Args:
        sampler: McSampler
            The handler that takes care of handling the weighted random sampling
        param_name:
            The parameter to generate the 1D PDF for
        binning:
            A one dimensional numpy array with the histogram binning
    """

    def __init__(self, sampler: McSampler, param_name: str, binning: np.ndarray):
        super(McPdf1D, self).__init__()
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
        samples = self.__sampler.getSamples()
        # For each object, take a random weighted sample and generate the histogram
        pdfs = np.zeros((len(samples), self.__binning.shape[0] - 1))

        for i in range(pdfs.shape[0]):
            pdfs[i, :] = np.histogram(
                samples[i][self.__param_name], bins=self.__binning, density=True
            )[0]

        return [
            Column(pdfs, 'MC_PDF_1D_{}'.format(self.__param_name.upper()))
        ]