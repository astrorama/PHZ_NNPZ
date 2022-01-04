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
Created on: 28/02/18
Author: Nikolaos Apostolakos
"""

import os.path

from ElementsKernel import Logging
from astropy.table import Table
from scipy import interpolate

from nnpz.config import ConfigManager
from nnpz.neighbor_selection.bruteforce import BruteForceSelector
from nnpz.neighbor_selection.combined import CombinedSelector
from nnpz.neighbor_selection.kdtree import KDTreeSelector
from nnpz.neighbor_selection.scaledbruteforce import ScaledBruteForceSelector

logger = Logging.getLogger('Configuration')


class NeighborSelectorConfig(ConfigManager.ConfigHandler):
    """
    Configure the search strategy for finding neighbors
    """

    def __init__(self):
        self.__selector = None
        self.__scaling = None
        self.__neighbors_no = None
        self.__ref_bands = None

    @staticmethod
    def __getScalePrior(prior):
        if hasattr(prior, '__call__'):
            return prior
        if prior == 'uniform':
            return lambda a: 1
        if os.path.exists(prior):
            table = Table.read(prior, format='ascii')
            return interpolate.interp1d(table.columns[0], table.columns[1], kind='linear')
        raise Exception('Unknown prior')

    def __createSelector(self, args):
        self._checkParameterExists('neighbor_method', args)
        self._checkParameterExists('neighbors_no', args)
        self._checkParameterExists('reference_sample_phot_filters', args)

        neighbor_method = args['neighbor_method']
        scale_prior = args.get('scale_prior', None)

        self.__neighbors_no = args['neighbors_no']
        self.__ref_bands = args['reference_sample_phot_filters']

        if neighbor_method not in ['KDTree', 'Combined', 'BruteForce']:
            raise ValueError('Invalid neighbor_method %s' % neighbor_method)
        if scale_prior and neighbor_method != 'BruteForce':
            raise ValueError('Scaling is only supported with BruteForce')

        logger.info('Using %s%s', neighbor_method, ' with scaling' if scale_prior else '')

        if neighbor_method == 'KDTree':
            self.__selector = KDTreeSelector(
                self.__neighbors_no, balanced=args.get('balanced_kdtree', True)
            )
        elif neighbor_method == 'Combined':
            self._checkParameterExists('batch_size', args)
            self.__selector = CombinedSelector(
                self.__neighbors_no, args['batch_size'],
                balanced=args.get('balanced_kdtree', True)
            )
        elif scale_prior:
            self._checkParameterExists('batch_size', args)
            self.__selector = ScaledBruteForceSelector(
                self.__neighbors_no, scale_prior, batch_size=args['batch_size'],
                max_iter=args.get('scale_max_iter', 20), rtol=args.get('scale_rtol', 1e-4)
            )
        else:
            self.__selector = BruteForceSelector(self.__neighbors_no)

    def parseArgs(self, args):
        if self.__selector is None:
            self.__createSelector(args)
        return {
            'neighbor_selector': self.__selector,
            'scaling': self.__scaling,
            'neighbor_no': self.__neighbors_no
        }


ConfigManager.addHandler(NeighborSelectorConfig)
