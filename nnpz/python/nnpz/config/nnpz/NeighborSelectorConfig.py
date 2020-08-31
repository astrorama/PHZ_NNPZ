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

"""
Created on: 28/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import os.path
import sys

import nnpz.neighbor_selection.brute_force_methods as bfm
from ElementsKernel import Logging
from astropy.table import Table
from nnpz.config import ConfigManager
from nnpz.config.nnpz import TargetCatalogConfig
from nnpz.neighbor_selection import (KDTreeSelector, BruteForceSelector,
                                     EuclideanRegionBruteForceSelector)
from nnpz.neighbor_selection.AdaptiveSelector import AdaptiveSelector
from nnpz.scaling import Chi2Scaling
from scipy import interpolate

logger = Logging.getLogger('Configuration')


class NeighborSelectorConfig(ConfigManager.ConfigHandler):
    """
    Configure the search strategy for finding neighbors
    """

    def __init__(self):
        self.__selector = None
        self.__scaling = None

    @staticmethod
    def __getPrior(prior):
        if hasattr(prior, '__call__'):
            return prior
        elif prior == 'uniform':
            return lambda a: 1
        elif os.path.exists(prior):
            table = Table.read(prior, format='ascii')
            return interpolate.interp1d(table.columns[0], table.columns[1], kind='linear')
        raise Exception('Unknown prior')

    def __createSelector(self, args):
        self._checkParameterExists('neighbor_method', args)
        neighbor_method = args['neighbor_method']
        self._checkParameterExists('neighbors_no', args)
        neighbors_no = args['neighbors_no']
        use_adaptive_bands = args.get('neighbor_adaptive_bands', False)

        if neighbor_method == 'KDTree':
            self.__selector = KDTreeSelector(
                neighbors_no, balanced_tree=args.get('balanced_kdtree', True)
            )
        elif neighbor_method == 'BruteForce':
            scale_prior = args.get('scale_prior', None)
            if scale_prior:
                self._checkParameterExists('batch_size', args)
                self.__scaling = Chi2Scaling(
                    self.__getPrior(args['scale_prior']),
                    batch_size=args.get('batch_size'),
                    max_iter=args.get('scale_max_iter', 20), rtol=args.get('scale_rtol', 1e-4)
                )
            self.__selector = BruteForceSelector(
                bfm.Chi2Distance(), bfm.SmallestSelector(neighbors_no), self.__scaling
            )
        elif neighbor_method == 'Combined':
            self._checkParameterExists('batch_size', args)
            self.__selector = EuclideanRegionBruteForceSelector(
                neighbors_no, args['batch_size'], balanced_tree=args.get('balanced_kdtree', True)
            )
        else:
            logger.error('Invalid neighbor_method option: %s', neighbor_method)
            sys.exit(-1)

        if use_adaptive_bands:
            target_config = ConfigManager.getHandler(TargetCatalogConfig).parseArgs(args)
            self.__selector = AdaptiveSelector(
                self.__selector, target_config['target_phot_data'], target_config['target_filters']
            )

    def parseArgs(self, args):
        if self.__selector is None:
            self.__createSelector(args)
        return {
            'neighbor_selector': self.__selector,
            'scaling': self.__scaling,
        }


ConfigManager.addHandler(NeighborSelectorConfig)