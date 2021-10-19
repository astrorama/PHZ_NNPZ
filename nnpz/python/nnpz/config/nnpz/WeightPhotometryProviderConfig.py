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
Created on: 11/04/18
Author: Alejandro Alvarez Ayllon
"""

from __future__ import division, print_function

import sys
from ElementsKernel import Logging
from nnpz.config import ConfigManager
from nnpz.config.nnpz import TargetCatalogConfig
from nnpz.config.reference import ReferenceConfig
from nnpz.weights import CopiedPhotometry, CorrectedPhotometry, RecomputedPhotometry

logger = Logging.getLogger('Configuration')


class WeightPhotometryProviderConfig(ConfigManager.ConfigHandler):
    """
    Configure the weighting strategy
    """

    def __init__(self):
        self.__photometry_provider = None

    def __createCopiedPhotometry(self, args):
        ref_config = ConfigManager.getHandler(ReferenceConfig).parseArgs(args)
        ref_phot = ref_config['reference_photometry']
        self.__photometry_provider = CopiedPhotometry(ref_phot)

    def __createRecomputedPhotometry(self, args):
        self._checkParameterExists('reference_sample_phot_filters', args)

        reference_config = ConfigManager.getHandler(ReferenceConfig).parseArgs(args)
        target_config = ConfigManager.getHandler(TargetCatalogConfig).parseArgs(args)

        if 'reference_sample' not in reference_config \
                or reference_config['reference_sample'] is None \
                or not hasattr(reference_config['reference_sample'], 'getSedData'):
            logger.error('CONFIGURATION ERROR:')
            logger.error('Target_catalog_gal_ebv and target_catalog_filters_mean are only '
                         'supported when reference_sample_dir is used')
            sys.exit(1)

        ref_sample = reference_config['reference_sample']
        ref_phot = reference_config['reference_photometry']
        phot_type = reference_config['reference_phot_type']
        ebv = target_config['target_ebv']
        trans_mean = target_config['target_filter_mean_wavelength']

        self.__photometry_provider = RecomputedPhotometry(
            ref_sample, ref_phot, phot_type=phot_type,
            ebv_list=ebv, filter_trans_mean_lists=trans_mean,
            oversample_filter=args.get('recomputed_filter_oversample', None),
            oversample_kind=args.get('recomputed_filter_oversample_type', 'linear')
        )

    def __createCorrectedPhotometry(self, args):
        self._checkParameterExists('reference_sample_phot_filters', args)

        reference_config = ConfigManager.getHandler(ReferenceConfig).parseArgs(args)
        target_config = ConfigManager.getHandler(TargetCatalogConfig).parseArgs(args)

        if 'reference_sample' not in reference_config \
                or reference_config['reference_sample'] is None \
                or not hasattr(reference_config['reference_sample'], 'getSedData'):
            logger.error('CONFIGURATION ERROR:')
            logger.error('Target_catalog_gal_ebv and target_catalog_filters_mean are only '
                         'supported when reference_sample_dir is used')
            sys.exit(1)

        ref_phot = reference_config['reference_photometry']
        ref_filters = reference_config['reference_filters']
        ebv = target_config['target_ebv']
        trans_mean = target_config['target_filter_mean_wavelength']

        self.__photometry_provider = CorrectedPhotometry(
            ref_phot, ref_filters, ebv_list=ebv, filter_trans_mean_lists=trans_mean
        )

    def __createPhotometryProvider(self, args):
        target_config = ConfigManager.getHandler(TargetCatalogConfig).parseArgs(args)
        if target_config['target_ebv'] is not None or target_config[
            'target_filter_mean_wavelength'] is not None:
            if target_config.get('recompute_photometry', False):
                logger.info('Using recomputed photometries for weight calculation')
                self.__createRecomputedPhotometry(args)
            else:
                logger.info('Using corrected photometries for weight calculation')
                self.__createCorrectedPhotometry(args)
        else:
            logger.info('Using copied photometries for weight calculation')
            self.__createCopiedPhotometry(args)

    def parseArgs(self, args):
        if self.__photometry_provider is None:
            self.__createPhotometryProvider(args)
        return {'photometry_provider': self.__photometry_provider}


ConfigManager.addHandler(WeightPhotometryProviderConfig)
