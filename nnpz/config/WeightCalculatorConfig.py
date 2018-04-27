"""
Created on: 26/04/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from nnpz.utils import Logging
from nnpz.config import ConfigManager
from nnpz.weights import (LikelihoodWeight, InverseEuclideanWeight, InverseChi2Weight)


logger = Logging.getLogger('Configuration')

_calculator_map = {
    'Euclidean' : InverseEuclideanWeight(),
    'Chi2' : InverseChi2Weight(),
    'Likelihood' : LikelihoodWeight()
}

class WeightCalculatorConfig(ConfigManager.ConfigHandler):

    def __init__(self):
        self.__calculator = None

    def __createCalculator(self, args):
        self._checkParameterExists('weight_method', args)
        method = args['weight_method']
        if not method in _calculator_map:
            logger.error('Unknown weight calculation method: {}'.format(method))
            exit(1)
        self.__calculator = _calculator_map[method]

        alternative = args.get('weight_method_alternative', None)
        self.__calculator_alternative = _calculator_map[alternative] if alternative else None

    def parseArgs(self, args):
        if self.__calculator is None:
            self.__createCalculator(args)
        return {
            'weight_calculator' : self.__calculator,
            'weight_calculator_alternative': self.__calculator_alternative
        }

ConfigManager.addHandler(WeightCalculatorConfig)
