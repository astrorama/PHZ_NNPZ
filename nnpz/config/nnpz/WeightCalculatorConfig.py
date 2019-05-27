"""
Created on: 26/04/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np
import nnpz.io.catalog_properties as prop
from nnpz.config import ConfigManager
from nnpz.config.nnpz import WeightPhotometryProviderConfig, ReferenceConfig
from nnpz.framework import ReferenceSampleWeightCalculator
from nnpz.framework.ReferenceSampleWeightCorrector import ReferenceSampleWeightCorrector
from nnpz.io import CatalogReader
from nnpz.utils import Logging
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

    def __setupAbsoluteWeights(self, args):
        ref_config = ConfigManager.getHandler(ReferenceConfig).parseArgs(args)

        # If there is a colon, the column is on a separate catalog
        column_name = args['absolute_weights']
        verify_ids = False
        if column_name.find(':') >= 0:
            catalog_path, column_name = column_name.split(':', 1)
            verify_ids = True
        # Otherwise, we look for this column on the reference sample or reference catalog
        elif ref_config.get('reference_catalog', None):
            catalog_path = ref_config['reference_catalog']
        else:
            catalog_path = ref_config['reference_sample_phot_file']

        logger.info('Using absolute weights "{}" from {}'.format(column_name, catalog_path))

        reader = CatalogReader(catalog_path)
        absolute_weights = reader.get(prop.Column(column_name))

        # If we are loading from a different catalog, make sure the IDs
        # match
        if verify_ids:
            weights_ids = reader.get(prop.ID)
            if np.any(weights_ids != ref_config['reference_ids']):
                logger.error('ERROR: Weight IDs do not match the reference IDs')

        self.__ref_sample_weight_calculator = ReferenceSampleWeightCorrector(
            self.__ref_sample_weight_calculator, absolute_weights
        )

    def __createCalculator(self, args):
        photo_provider_config = ConfigManager.getHandler(WeightPhotometryProviderConfig).parseArgs(args)

        self._checkParameterExists('weight_method', args)
        method = args['weight_method']
        if not method in _calculator_map:
            logger.error('Unknown weight calculation method: {}'.format(method))
            exit(1)
        self.__calculator = _calculator_map[method]

        alternative = args.get('weight_method_alternative', None)
        self.__calculator_alternative = _calculator_map[alternative] if alternative else None

        self.__ref_sample_weight_calculator = ReferenceSampleWeightCalculator(
            photo_provider_config['photometry_provider'], self.__calculator, self.__calculator_alternative
        )

        if args.get('absolute_weights', None):
            self.__setupAbsoluteWeights(args)

    def parseArgs(self, args):
        if self.__calculator is None:
            self.__createCalculator(args)
        return {
            'weight_calculator' : self.__ref_sample_weight_calculator,
        }

ConfigManager.addHandler(WeightCalculatorConfig)
