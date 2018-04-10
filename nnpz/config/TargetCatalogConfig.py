"""
Created on: 28/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from nnpz.utils import Logging
from nnpz.config import ConfigManager
from nnpz.io import CatalogReader
import nnpz.io.catalog_properties as prop

logger = Logging.getLogger('Configuration')


class TargetCatalogConfig(ConfigManager.ConfigHandler):


    def __init__(self):
        self.__target_ids = None
        self.__target_phot_data = None
        self.__target_astropy_table = None


    def __createData(self, args):

        self._checkParameterExists('target_catalog', args)
        target_cat = args['target_catalog']
        self._checkParameterExists('target_catalog_filters', args)
        target_filters = args['target_catalog_filters']
        missing_phot_flags = args.get('missing_photometry_flags', [])

        logger.info('Reading target catalog: {}'.format(target_cat))
        target_reader = CatalogReader(target_cat)
        self.__target_ids = target_reader.get(prop.ID)
        self.__target_phot_data =target_reader.get(prop.Photometry(target_filters, missing_phot_flags))
        self.__target_astropy_table = target_reader.getAsAstropyTable()

        if 'input_size' in args:
            input_size = args['input_size']
            logger.warn('Processing only first {} objects from target catalog'.format(input_size))
            self.__target_ids = self.__target_ids[:input_size]
            self.__target_phot_data = self.__target_phot_data[:input_size]
            self.__target_astropy_table = self.__target_astropy_table[:input_size]


    def parseArgs(self, args):

        if self.__target_ids is None:
            self.__createData(args)

        return {'target_ids' : self.__target_ids,
                'target_phot_data' : self.__target_phot_data,
                'target_astropy_table' : self.__target_astropy_table}


ConfigManager.addHandler(TargetCatalogConfig)