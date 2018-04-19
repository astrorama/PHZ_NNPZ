"""
Created on: 11/04/18
Author: Alejandro Alvarez Ayllon
"""

from __future__ import division, print_function

from nnpz.config import ConfigManager, ReferenceSamplePhotometryConfig, ReferenceSampleConfig, TargetCatalogConfig
from nnpz.utils import Logging
from nnpz.weights import CopiedPhotometry, RecomputedPhotometry

logger = Logging.getLogger('Configuration')


class WeightPhotometryProviderConfig(ConfigManager.ConfigHandler):

    def __init__(self):
        self.__photometry_provider = None

    def __createCopiedPhotometry(self, args):
        ref_phot_data = ConfigManager.getHandler(ReferenceSamplePhotometryConfig).parseArgs(args)['reference_phot_data']

        self.__photometry_provider = CopiedPhotometry(ref_phot_data)

    def __createRecomputedPhotometry(self, args):
        self._checkParameterExists('reference_sample_phot_filters', args)

        ref_sample_photometry_config = ConfigManager.getHandler(ReferenceSamplePhotometryConfig).parseArgs(args)
        ref_sample_config = ConfigManager.getHandler(ReferenceSampleConfig).parseArgs(args)
        target_config = ConfigManager.getHandler(TargetCatalogConfig).parseArgs(args)

        ref_sample = ref_sample_config['reference_sample']
        filter_order = args['reference_sample_phot_filters']
        filter_trans_map = ref_sample_photometry_config['reference_filter_transmission']
        phot_type = ref_sample_photometry_config['reference_phot_type']
        ebv = target_config['target_ebv']
        trans_mean = target_config['target_filter_mean_wavelength']

        self.__photometry_provider = RecomputedPhotometry(
            ref_sample, filter_order, filter_trans_map, phot_type,
            ebv_list=ebv, filter_trans_mean_lists=trans_mean
        )

    def __createPhotometryProvider(self, args):
        if args.get('target_ebv', None) or args.get('target_filter_mean_wavelength', None):
            logger.info('Using recomputed photometries for weight calculation')
            self.__createRecomputedPhotometry(args)
        else:
            logger.info('Using copied photometries for weight calculation')
            self.__createCopiedPhotometry(args)


    def parseArgs(self, args):
        if self.__photometry_provider is None:
            self.__createPhotometryProvider(args)
        return {'photometry_provider': self.__photometry_provider}


ConfigManager.addHandler(WeightPhotometryProviderConfig)
