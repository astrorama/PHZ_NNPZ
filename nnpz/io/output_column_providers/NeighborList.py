"""
Created on: 02/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from astropy.table import Column

from nnpz.io import OutputHandler


class NeighborList(OutputHandler.OutputColumnProviderInterface):

    def __init__(self, catalog_size, ref_ids):
        self.__neighbors = [[] for i in range(catalog_size)]
        self.__weights = [[] for i in range(catalog_size)]
        self.__ref_ids = ref_ids

    def addContribution(self, reference_sample_i, catalog_i, weight, flags):
        self.__neighbors[catalog_i].append(self.__ref_ids[reference_sample_i])
        self.__weights[catalog_i].append(weight)

    def getColumns(self):
        neighbor_col = Column(self.__neighbors, 'NeighborIDs')
        weight_col = Column(self.__weights, 'NeighborWeights')
        return [neighbor_col, weight_col]

