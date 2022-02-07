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
from typing import List, Optional, Tuple

import numpy as np
from ElementsKernel import Logging
from astropy import units as u
from nnpz.io import OutputHandler
from nnpz.reference_sample.MontecarloProvider import MontecarloProvider

logger = Logging.getLogger(__name__)


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

    def __init__(self, take_n: int, mc_provider: MontecarloProvider, ref_ids: np.ndarray):
        self.__take_n = take_n
        self.__provider = mc_provider
        self.__ref_ids = ref_ids
        self.__samples = None
        self.__dtype = self.__provider.getDtype()
        self.__samples_per_neighbor = self.__provider.getNSamples()
        self.__rng = np.random.default_rng()

    def get_provider(self):
        return self.__provider

    def get_column_definition(self) \
            -> List[Tuple[str, np.dtype, u.Unit, Optional[Tuple[int, ...]]]]:
        """
        This provider does not generate any output
        """
        return []

    def generate_output(self, indexes: np.ndarray, neighbor_info: np.ndarray, output: np.ndarray):
        """
        This provider does not generate any output, but hooks this call to generate the samples
        """
        logger.info('Sampling physical parameters')
        n_neighbors = neighbor_info['NEIGHBOR_WEIGHTS'].shape[1]
        # Now, for each object we need 'take_n' samples from a bag composed of
        # 'samples_per_neighbor' * 'neighbor_no' measurements. All the samples contributed by a
        # given neighbor have the same weight
        # To avoid reading everything and just selecting a subset, we actually sample from
        # (object-index, sample-index), and then we read those selected.
        # To improve i/o performance, we read *all* selected samples for *all* objects. This allows
        # the provider to do the reads from disk sequentially
        selected_indexes = np.full((len(indexes), self.__take_n, 2), fill_value=-1, dtype=int)
        bag = np.arange(0, self.__samples_per_neighbor * n_neighbors, dtype=int)
        for i in range(len(indexes)):
            neighbors = neighbor_info[i]
            p = np.repeat(neighbors['NEIGHBOR_WEIGHTS'], repeats=self.__samples_per_neighbor)
            total_weight = p.sum()
            # If all neighbors have 0 weight, we can not pick any
            if total_weight <= 0 or not np.isfinite(total_weight):
                continue
            # Pick the sample indexes
            p /= total_weight
            selected = self.__rng.choice(bag, size=self.__take_n, replace=True, p=p)
            np.divmod(selected, self.__samples_per_neighbor, selected_indexes[i, :, 0],
                      selected_indexes[i, :, 1])
            selected_indexes[i, :, 0] = neighbors['NEIGHBOR_INDEX'][selected_indexes[i, :, 0]]
        # Defer to the provider to read the samples
        self.__samples = self.__provider.getDataForIndex(selected_indexes.reshape(-1, 2))
        self.__samples = self.__samples.reshape((len(indexes), self.__take_n))

    def get_samples(self):
        """
        Get the sampling for the whole chunk
        """
        return self.__samples
