"""
Created on: 02/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np
from astropy.table import Column

from nnpz.io import OutputHandler


class NeighborList(OutputHandler.OutputColumnProviderInterface):

    def __init__(self, catalog_size, ref_ids, n_neighbors):
        self.__neighbors = np.zeros((catalog_size, n_neighbors), dtype=np.int64)
        self.__weights = np.zeros((catalog_size, n_neighbors), dtype=np.float32)
        self.__offset = np.zeros(catalog_size, dtype=np.uint32)
        self.__ref_ids = ref_ids

    def addContribution(self, reference_sample_i, catalog_i, weight, flags):
        offset = self.__offset[catalog_i]
        self.__neighbors[catalog_i, offset] = self.__ref_ids[reference_sample_i]
        self.__weights[catalog_i, offset] = weight
        self.__offset[catalog_i] += 1

    def getColumns(self):
        neighbor_col = Column(self.__neighbors, 'NeighborIDs')
        weight_col = Column(self.__weights, 'NeighborWeights')
        return [neighbor_col, weight_col]
