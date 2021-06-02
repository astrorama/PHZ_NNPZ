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
from nnpz.io import OutputHandler
from nnpz.reference_sample.MontecarloProvider import MontecarloProvider


class McSampler(OutputHandler.OutputColumnProviderInterface):
    """
    For each target object, joins *all* the n-dimensional samples from its neighbors,
    and randomly re-sample this set using the corresponding weights
    i.e. All samples belonging to a neighbor with weight 0.5 will have a weight of 0.5. This is,
    they will be all equally likely

    Args:
        catalog_size:
            Size of the target catalog
        n_neighbors:
            Number of neighbors per object
        take_n:
            How many samples to take from the union
        mc_provider:
            MontecarloProvider with the samples from the n-dimensional space
        ref_ids:
            Reference object ids

    Notes:
        The samples are generated only once, so multiple PDF will share the selection.
        They are generated *at the end*, since we need to know all the weights before we can do
        the sampling.
        Unlike other parts of the code, the data from the reference sample is accessed at the
        end, and following the order of the target catalog. The disk access will be less
        predictable, but keeping in memory all the samples until the end can be quite memory
        consuming.
    """

    def __init__(self, catalog_size: int, n_neighbors: int, take_n: int,
                 mc_provider: MontecarloProvider, ref_ids: np.ndarray):
        self.__samples_per_neighbor = mc_provider.getData(ref_ids[0]).shape[0]
        self.__neighbor_ids = np.zeros((catalog_size, n_neighbors), dtype=np.int32)
        self.__neighbor_weights = np.zeros((catalog_size, n_neighbors), dtype=np.float32)
        self.__take_n = take_n
        self.__provider = mc_provider
        self.__ref_ids = ref_ids
        self.__samples = None

    def getProvider(self):
        """
        Return the provider
        """
        return self.__provider

    def getColumnDefinition(self):
        """
        This provider does not generate any output
        """
        return []

    def setWriteableArea(self, output_area):
        """
        This provider does not generate any output
        """
        pass

    def addContribution(self, reference_sample_i, neighbor, flags):
        """
        See OutputColumnProviderInterface.addContribution
        """
        self.__neighbor_ids[neighbor.index, neighbor.position] = self.__ref_ids[reference_sample_i]
        self.__neighbor_weights[neighbor.index, neighbor.position] = neighbor.weight

    def generateSamples(self, index):
        """
        Generated the weighted random sample for the target object i
        """
        nb_sample_weight = np.repeat(self.__neighbor_weights[index], self.__samples_per_neighbor)
        nb_total_weight = nb_sample_weight.sum()

        # If all neighbors have 0 weight, just fill with nan or 0 (depending on the type)
        if nb_total_weight <= 0:
            prototype = self.__provider.getData(self.__neighbor_ids[0, 0])
            prototype = np.repeat(prototype, len(self.__neighbor_ids[index]))
            return np.zeros_like(prototype[:self.__take_n])

        nb_sample_weight /= nb_total_weight

        nb_samples = [self.__provider.getData(nid) for nid in self.__neighbor_ids[index]]
        nb_samples = np.concatenate(nb_samples)
        chosen = np.random.choice(nb_samples.shape[0], size=self.__take_n, p=nb_sample_weight)
        return nb_samples[chosen]

    def getSamples(self):
        """
        Get the sampling for the whole target catalog
        """
        if self.__samples is None:
            samples = list()
            for i in range(self.__neighbor_ids.shape[0]):
                samples.append(self.generateSamples(i))
            self.__samples = np.stack(samples)
        return self.__samples

    def fillColumns(self):
        """
        This provider does not generate any output
        """
        return []

    def getSampleCount(self):
        return self.__take_n

    def getDtype(self, name):
        return self.__provider.getDtype(name)
