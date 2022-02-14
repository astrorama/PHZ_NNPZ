#
# Copyright (C) 2012-2022 Euclid Science Ground Segment
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
from typing import Any, Dict

from ElementsKernel import Logging
from nnpz.config import ConfigManager
from nnpz.config.neighbors.Scaling import Scaling
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
        self.__scaling = False
        self.__neighbors_no = None
        self.__ref_bands = None

    def __create_selector(self, args: Dict[str, Any]):
        self._exists_parameter('neighbor_method', args)
        self._exists_parameter('neighbors_no', args)
        self._exists_parameter('reference_sample_phot_filters', args)

        neighbor_method = args['neighbor_method']

        scaler = ConfigManager.get_handler(Scaling).parse_args(args)['scaler']

        self.__neighbors_no = args['neighbors_no']
        self.__ref_bands = args['reference_sample_phot_filters']

        if neighbor_method not in ['KDTree', 'Combined', 'BruteForce']:
            raise ValueError('Invalid neighbor_method %s' % neighbor_method)
        if scaler is not None and neighbor_method != 'BruteForce':
            raise ValueError('Scaling is only supported with BruteForce')

        logger.info('Using %s%s', neighbor_method, ' with scaling' if scaler else '')

        if neighbor_method == 'KDTree':
            self.__selector = KDTreeSelector(
                self.__neighbors_no, balanced=args.get('balanced_kdtree', True)
            )
        elif neighbor_method == 'Combined':
            self._exists_parameter('batch_size', args)
            self.__selector = CombinedSelector(
                self.__neighbors_no, args['batch_size'],
                balanced=args.get('balanced_kdtree', True)
            )
        elif scaler:
            self.__selector = ScaledBruteForceSelector(self.__neighbors_no, scaler)
        else:
            self.__selector = BruteForceSelector(self.__neighbors_no)

    def parse_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if self.__selector is None:
            self.__create_selector(args)
        return {
            'neighbor_selector': self.__selector,
            'neighbor_no': self.__neighbors_no
        }


ConfigManager.add_handler(NeighborSelectorConfig)
