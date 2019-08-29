"""
Created on: 28/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import nnpz.io.catalog_properties as prop
from ElementsKernel import Logging
from nnpz.config import ConfigManager
from nnpz.io import CatalogReader

logger = Logging.getLogger('Configuration')


class TargetCatalogConfig(ConfigManager.ConfigHandler):


    def __init__(self):
        self.__target_size = None
        self.__target_id_column = None
        self.__target_ids = None
        self.__target_phot_data = None
        self.__target_astropy_table = None
        self.__target_ebv = None
        self.__target_filter_mean_wavelength = None


    def __createData(self, args):

        self._checkParameterExists('target_catalog', args)
        target_cat = args['target_catalog']
        self._checkParameterExists('target_catalog_filters', args)
        target_filters = args['target_catalog_filters']
        missing_phot_flags = args.get('missing_photometry_flags', [])
        self.__target_id_column = args.get('target_catalog_id_column', 'ID')

        logger.info('Target catalog photometric columns: {}'.format(target_filters))

        logger.info('Reading target catalog: {}'.format(target_cat))
        target_reader = CatalogReader(target_cat)
        self.__target_ids = target_reader.get(prop.ID(self.__target_id_column))
        self.__target_phot_data = target_reader.get(prop.Photometry(target_filters, missing_phot_flags))
        self.__target_astropy_table = target_reader.getAsAstropyTable()

        target_catalog_ebv = args.get('target_catalog_gal_ebv', None)
        if target_catalog_ebv is not None:
            logger.info('Using E(B-V) columns {}'.format(target_catalog_ebv))
            self.__target_ebv = target_reader.get(prop.EBV(target_catalog_ebv, nan_flags=missing_phot_flags))

        target_catalog_filters_mean = args.get('target_catalog_filters_mean', None)
        if target_catalog_filters_mean is not None:
            logger.info('Using filters mean wavelength columns {}'.format(target_catalog_filters_mean))
            self.__target_filter_mean_wavelength = target_reader.get(
                prop.FiltersMeanWavelength(target_catalog_filters_mean, missing_phot_flags)
            )

        if 'input_size' in args:
            input_size = args['input_size']
            logger.warn('Processing only first {} objects from target catalog'.format(input_size))
            self.__target_ids = self.__target_ids[:input_size]
            self.__target_phot_data = self.__target_phot_data[:input_size]
            self.__target_astropy_table = self.__target_astropy_table[:input_size]

        self.__target_size = len(self.__target_ids)


    def parseArgs(self, args):

        if self.__target_ids is None:
            self.__createData(args)

        return {'target_id_column': self.__target_id_column,
                'target_ids' : self.__target_ids,
                'target_phot_data' : self.__target_phot_data,
                'target_ebv': self.__target_ebv,
                'target_filter_mean_wavelength': self.__target_filter_mean_wavelength,
                'target_astropy_table' : self.__target_astropy_table,
                'target_size' : self.__target_size}


ConfigManager.addHandler(TargetCatalogConfig)
