"""
Created on: 28/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np

from nnpz.utils import Logging
from nnpz.config import (ConfigManager, ReferenceSampleConfig)
from nnpz.photometry import PhotometryProvider

logger = Logging.getLogger('Configuration')


class ReferenceSamplePhotometryConfig(ConfigManager.ConfigHandler):


    def __init__(self):
        self.__ref_phot_data = None
        self.__out_mean_phot_filters = None
        self.__out_mean_phot_data = None


    def __createData(self, args):

        ref_sample_dict = ConfigManager.getHandler(ReferenceSampleConfig).parseArgs(args)
        if 'reference_sample' in ref_sample_dict:

            self._checkParameterExists('reference_sample_phot_file', args)
            phot_file = args['reference_sample_phot_file']
            self._checkParameterExists('reference_sample_phot_filters', args)
            phot_filters = args['reference_sample_phot_filters']

            logger.info('Using reference sample photometry from {}'.format(phot_file))
            ref_phot_prov = PhotometryProvider(phot_file)
            if np.any(ref_phot_prov.getIds() != ref_sample_dict['reference_ids']):
                logger.error('ERROR: Reference sample photometry ID mismatch')
                exit(-1)

            logger.info('Reference sample photometric bands: {}'.format(phot_filters))
            self.__ref_phot_data = ref_phot_prov.getData(*phot_filters)

            if 'reference_sample_out_mean_phot_filters' in args:
                self.__out_mean_phot_filters = args['reference_sample_out_mean_phot_filters']
                self.__out_mean_phot_data = ref_phot_prov.getData(*self.__out_mean_phot_filters)


    def parseArgs(self, args):

        if self.__ref_phot_data is None and self.__out_mean_phot_data is None:
            self.__createData(args)

        result = {}
        if self.__ref_phot_data is not None:
            result['reference_phot_data'] = self.__ref_phot_data
        if self.__out_mean_phot_filters is not None:
            result['out_mean_phot_filters'] = self.__out_mean_phot_filters
            result['out_mean_phot_data'] = self.__out_mean_phot_data
        return result


ConfigManager.addHandler(ReferenceSamplePhotometryConfig)