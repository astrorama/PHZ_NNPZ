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

"""
Created on: 27/05/19
Author: Alejandro Alvarez Ayllon
"""

from __future__ import division, print_function

import numpy as np
from ElementsKernel import Logging
from nnpz.config import ConfigManager
from nnpz.exceptions import UnknownNameException
from nnpz.io import CatalogReader
from nnpz.io.catalog_properties import Column
from nnpz.io.output_column_providers.NeighborList import NEIGHBOR_IDS_COLNAME, \
    NEIGHBOR_SCALES_COLNAME, NEIGHBOR_WEIGHTS_COLNAME

logger = Logging.getLogger('Configuration')


class NnpzCatalogConfig(ConfigManager.ConfigHandler):
    """
    Configure the NNPZ output catalog, which here is used as an input.
    """

    def __init__(self):
        self.__added = False
        self.__neighbors = None
        self.__weights = None
        self.__idxs = None
        self.__astropy_table = None
        self.__scales = None

    def _loadNnpzCatalog(self, args):
        self._checkParameterExists('nnpz_catalog', args)
        nnpz_cat = args['nnpz_catalog']

        logger.info('Reading NNPZ catalog: %s', nnpz_cat)
        nnpz_reader = CatalogReader(nnpz_cat)
        self.__neighbors = nnpz_reader.get(Column(NEIGHBOR_IDS_COLNAME, dtype=np.int64))
        self.__weights = nnpz_reader.get(Column(NEIGHBOR_WEIGHTS_COLNAME, dtype=np.float64))
        try:
            self.__scales = nnpz_reader.get(Column(NEIGHBOR_SCALES_COLNAME, dtype=np.float64))
        except UnknownNameException:
            self.__scales = np.ones(self.__weights.shape)
        self.__idxs = np.arange(len(self.__neighbors))
        self.__astropy_table = nnpz_reader.getAsAstropyTable()

        if 'input_size' in args:
            input_size = args['input_size']
            if not isinstance(input_size, slice):
                input_size = slice(input_size)
            self.__neighbors = self.__neighbors[input_size]
            self.__weights = self.__weights[input_size]
            self.__scales = self.__scales[input_size]
            self.__idxs = self.__idxs[input_size]

    def parseArgs(self, args):
        if not self.__added:
            self._loadNnpzCatalog(args)
            self.__added = True

        return {
            'nnpz_idx': self.__idxs,
            'nnpz_neighbors': self.__neighbors,
            'nnpz_weights': self.__weights,
            'nnpz_scales': self.__scales,
            'nnpz_size': len(self.__idxs),
            'nnpz_astropy_table': self.__astropy_table
        }


ConfigManager.addHandler(NnpzCatalogConfig)
