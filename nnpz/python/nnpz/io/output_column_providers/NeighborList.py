"""
Created on: 02/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from astropy.table import Column

from nnpz.io import OutputHandler

NEIGHBOR_IDS_COLNAME = 'NEIGHBOR_IDS'
NEIGHBOR_WEIGHTS_COLNAME = 'NEIGHBOR_WEIGHTS'
NEIGHBOR_SCALES_COLNAME = 'NEIGHBOR_SCALING'


class NeighborList(OutputHandler.OutputColumnProviderInterface):

    def __init__(self, catalog_size, ref_ids):
        self.__neighbors = [[] for i in range(catalog_size)]
        self.__weights = [[] for i in range(catalog_size)]
        self.__scales = [[] for i in range(catalog_size)]
        self.__ref_ids = ref_ids

    def addContribution(self, reference_sample_i, neighbor, flags):
        self.__neighbors[neighbor.index].append(self.__ref_ids[reference_sample_i])
        self.__weights[neighbor.index].append(neighbor.weight)
        self.__scales[neighbor.index].append(neighbor.scale)

    def getColumns(self):
        neighbor_col = Column(self.__neighbors, NEIGHBOR_IDS_COLNAME)
        weight_col = Column(self.__weights, NEIGHBOR_WEIGHTS_COLNAME)
        scale_col = Column(self.__scales, NEIGHBOR_SCALES_COLNAME)
        return [neighbor_col, weight_col, scale_col]
