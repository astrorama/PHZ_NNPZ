"""
Created on: 28/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from nnpz.utils import Logging
from nnpz.config import ConfigManager
from nnpz.neighbor_selection import (KDTreeSelector, BruteForceSelector,
                                     EuclideanRegionBruteForceSelector)
import nnpz.neighbor_selection.brute_force_methods as bfm

logger = Logging.getLogger('Configuration')


class NeighborSelectorConfig(ConfigManager.ConfigHandler):

    def __init__(self):
        self.__selector = None

    def __createSelector(self, args):

        self._checkParameterExists('neighbor_method', args)
        neighbor_method = args['neighbor_method']
        self._checkParameterExists('neighbors_no', args)
        neighbors_no = args['neighbors_no']

        if neighbor_method == 'KDTree':
            self.__selector = KDTreeSelector(
                neighbors_no, balanced_tree=args.get('balanced_kdtree', True)
            )
        elif neighbor_method == 'BruteForce':
            self.__selector = BruteForceSelector(
                bfm.Chi2Distance(), bfm.SmallestSelector(neighbors_no)
            )
        elif neighbor_method == 'Combined':
            self._checkParameterExists('batch_size', args)
            self.__selector = EuclideanRegionBruteForceSelector(
                neighbors_no, args['batch_size'], balanced_tree=args.get('balanced_kdtree', True)
            )
        else:
            logger.error('Invalid neighbor_method option: {}'.format(neighbor_method))
            exit(-1)


    def parseArgs(self, args):
        if self.__selector is None:
            self.__createSelector(args)
        return {'neighbor_selector' : self.__selector}


ConfigManager.addHandler(NeighborSelectorConfig)