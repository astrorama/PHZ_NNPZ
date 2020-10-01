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
from nnpz.reference_sample.MontecarloProvider import MontecarloProvider


class McPdf1D(OutputHandler.OutputColumnProviderInterface):
    """
    Generate a one dimensional PDF for a given parameter using a weighted random sample
    from the reference objects nearest to a given object

    See Also:
        nnpz.io.output_hdul_providers.McPdf1DBins
    Args:
        catalog_size:
            Input catalog size
        n_neighbors:
            How many neighbors will be used
        take_n:
            Total number of samples to generate
        param_name:
            The parameter to generate the 1D PDF for
        binning:
            A one dimensional numpy array with the histogram binning
        mc_provider:
            The data provider
        ref_ids:
            Reference object IDs
    """

    def __init__(self, catalog_size: int, n_neighbors: int, take_n: int,
                 param_name: str, binning: np.ndarray,
                 mc_provider: MontecarloProvider, ref_ids: np.ndarray):
        super(McPdf1D, self).__init__()
        samples_per_neighbor = mc_provider.getData(ref_ids[0])[param_name].shape[0]
        self.__ref_ids = ref_ids
        self.__take_n = take_n
        self.__param_name = param_name
        self.__binning = binning
        self.__provider = mc_provider
        self.__current_ref_i = None
        self.__current_sample = None
        self.__samples = np.zeros((catalog_size, n_neighbors, samples_per_neighbor))
        self.__weights = np.zeros((catalog_size, n_neighbors, samples_per_neighbor))
        self.__obj_sample_idx = np.zeros(catalog_size, dtype=np.int)

    def addContribution(self, reference_sample_i, neighbor, flags):
        if reference_sample_i != self.__current_ref_i:
            current_sample = self.__provider.getData(self.__ref_ids[reference_sample_i])
            self.__current_sample = current_sample[self.__param_name]
            self.__current_ref_i = reference_sample_i

        sample_idx = self.__obj_sample_idx[neighbor.index]

        self.__samples[neighbor.index, sample_idx, :] = self.__current_sample[:]
        self.__weights[neighbor.index, sample_idx, :] = neighbor.weight

        self.__obj_sample_idx[neighbor.index] += 1

    def getColumns(self):
        # For each object, take a random weighted sample and generate the histogram
        pdfs = np.zeros((self.__samples.shape[0], self.__binning.shape[0] - 1))

        for i in range(pdfs.shape[0]):
            weights = self.__weights[i].reshape(-1)
            references = self.__samples[i].reshape(-1)
            weights /= weights.sum()
            samples = np.random.choice(references, size=self.__take_n, p=weights)
            pdfs[i, :] = np.histogram(samples, bins=self.__binning, density=True)[0]

        return [
            Column(pdfs, 'MC_PDF_1D_{}'.format(self.__param_name.upper()))
        ]
