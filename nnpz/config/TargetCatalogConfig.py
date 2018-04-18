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
        self.__target_ebv = None
        self.__target_filter_transmission = None


    def __createData(self, args):

        self._checkParameterExists('target_catalog', args)
        target_cat = args['target_catalog']
        self._checkParameterExists('target_catalog_filters', args)
        target_filters = args['target_catalog_filters']
        missing_phot_flags = args.get('missing_photometry_flags', [])

        logger.info('Reading target catalog: {}'.format(target_cat))
        target_reader = CatalogReader(target_cat)
        self.__target_ids = target_reader.get(prop.ID)
        self.__target_phot_data = target_reader.get(prop.Photometry(target_filters, missing_phot_flags))
        self.__target_astropy_table = target_reader.getAsAstropyTable()

        target_catalog_ebv = args.get('target_catalog_ebv', None)
        if target_catalog_ebv is not None:
            if not isinstance(target_catalog_ebv, tuple) and not isinstance(target_catalog_ebv, list):
                logger.error('target_catalog_ebv can only be a tuple or a list')
                exit(1)
            if len(target_catalog_ebv) != 2:
                logger.error('target_catalog_ebv must have length 2')
                exit(1)
            self.__target_ebv = target_reader.get(prop.EBV(*target_catalog_ebv, nan_flags=missing_phot_flags))

        target_catalog_filters_transmission = args.get('target_catalog_filters_transmission', None)
        if target_catalog_filters_transmission is not None:
            self.__target_filter_transmission = target_reader.get(
                prop.FiltersMeanWavelength(target_catalog_filters_transmission, missing_phot_flags)
            )

        if 'input_size' in args:
            input_size = args['input_size']
            logger.warn('Processing only first {} objects from target catalog'.format(input_size))
            self.__target_ids = self.__target_ids[:input_size]
            self.__target_phot_data = self.__target_phot_data[:input_size]
            self.__target_astropy_table = self.__target_astropy_table[:input_size]
            if self.__target_ebv is not None:
                self.__target_ebv = self.__target_ebv[:input_size]
            if self.__target_filter_transmission is not None:
                self.__target_filter_transmission = self.__target_filter_transmission[:input_size]


    def parseArgs(self, args):

        if self.__target_ids is None:
            self.__createData(args)

        return {'target_ids' : self.__target_ids,
                'target_phot_data' : self.__target_phot_data,
                'target_ebv': self.__target_ebv,
                'target_filter_transmission': self.__target_filter_transmission,
                'target_astropy_table' : self.__target_astropy_table}


ConfigManager.addHandler(TargetCatalogConfig)