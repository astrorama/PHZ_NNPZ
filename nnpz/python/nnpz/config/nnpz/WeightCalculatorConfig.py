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
Created on: 26/04/18
Author: Nikolaos Apostolakos
"""


import nnpz.io.catalog_properties as prop
import numpy as np
from ElementsKernel import Logging
from nnpz.config import ConfigManager
from nnpz.config.nnpz import NeighborSelectorConfig, WeightPhotometryProviderConfig
from nnpz.config.reference import ReferenceConfig
from nnpz.framework import ReferenceSampleWeightCalculator
from nnpz.framework.ReferenceSampleWeightCorrector import ReferenceSampleWeightCorrector
from nnpz.io import CatalogReader
from nnpz.weights import (InverseChi2Weight, InverseEuclideanWeight, LikelihoodWeight)

logger = Logging.getLogger('Configuration')

_calculator_map = {
    'Euclidean': InverseEuclideanWeight(),
    'Chi2': InverseChi2Weight(),
    'Likelihood': LikelihoodWeight()
}


class WeightCalculatorConfig(ConfigManager.ConfigHandler):

    def __init__(self):
        self.__ref_sample_weight_calculator = None
        self.__calculator = None
        self.__calculator_alternative = None

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

        logger.info('Using absolute weights "%s" from %s', column_name, catalog_path)

        reader = CatalogReader(catalog_path)
        absolute_weights = reader.get(prop.Column(column_name))

        # If we are loading from a different catalog, make sure the IDs  match
        if verify_ids:
            weights_ids = reader.get(prop.ID)
            if np.any(weights_ids != ref_config['reference_ids']):
                logger.error('ERROR: Weight IDs do not match the reference IDs')

        self.__ref_sample_weight_calculator = ReferenceSampleWeightCorrector(
            self.__ref_sample_weight_calculator, absolute_weights
        )

    def __createCalculator(self, args):
        photo_provider_config = ConfigManager.getHandler(WeightPhotometryProviderConfig).parseArgs(
            args)
        neighbor_sel_config = ConfigManager.getHandler(NeighborSelectorConfig).parseArgs(args)
        ref_config = ConfigManager.getHandler(ReferenceConfig).parseArgs(args)

        self._checkParameterExists('weight_method', args)
        method = args['weight_method']
        if not method in _calculator_map:
            logger.error('Unknown weight calculation method: %s', method)
            exit(1)
        self.__calculator = _calculator_map[method]

        alternative = args.get('weight_method_alternative', None)
        self.__calculator_alternative = _calculator_map[alternative] if alternative else None

        ref_filters = ref_config['reference_filters']
        self.__ref_sample_weight_calculator = ReferenceSampleWeightCalculator(
            photo_provider_config['photometry_provider'],
            self.__calculator, self.__calculator_alternative,
            ref_filters, scaling=neighbor_sel_config['scaling']
        )

        if args.get('absolute_weights', None):
            self.__setupAbsoluteWeights(args)

    def parseArgs(self, args):
        if self.__calculator is None:
            self.__createCalculator(args)
        return {
            'weight_calculator': self.__ref_sample_weight_calculator,
        }


ConfigManager.addHandler(WeightCalculatorConfig)
