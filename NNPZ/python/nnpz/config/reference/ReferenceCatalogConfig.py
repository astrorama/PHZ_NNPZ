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


class ReferenceCatalogConfig(ConfigManager.ConfigHandler):

    def __init__(self):
        self.__ref_ids = None
        self.__ref_phot_data = None
        self.__ref_z = None
        self.__out_mean_phot_filters = None
        self.__out_mean_phot_data = None
        self.__ref_cat = None

    def __createData(self, args):
        if 'reference_catalog' in args:
            self.__ref_cat = args['reference_catalog']
            self._checkParameterExists('reference_catalog_filters', args)
            ref_filters = args['reference_catalog_filters']
            self._checkParameterExists('reference_catalog_redshift', args)
            ref_z_col = args['reference_catalog_redshift']

            logger.info('Reading reference catalog from {}...'.format(self.__ref_cat))
            ref_reader = CatalogReader(self.__ref_cat)
            self.__ref_ids = ref_reader.get(prop.ID)
            self.__ref_phot_data = ref_reader.get(prop.Photometry(ref_filters))
            self.__ref_z = ref_reader.get(prop.Column(ref_z_col))

            if 'reference_catalog_out_mean_phot_filters' in args:
                out_filters = args['reference_catalog_out_mean_phot_filters']
                self.__out_mean_phot_filters = [f[0] for f in out_filters]
                self.__out_mean_phot_data = ref_reader.get(prop.Photometry(out_filters))


    def parseArgs(self, args):

        if self.__ref_ids is None:
            self.__createData(args)

        result = {}
        if self.__ref_ids is not None:
            result['reference_catalog'] = self.__ref_cat
            result['reference_ids'] = self.__ref_ids
            result['reference_phot_data'] = self.__ref_phot_data
            result['reference_redshift'] = self.__ref_z
        if self.__out_mean_phot_filters is not None:
            result['out_mean_phot_filters'] = self.__out_mean_phot_filters
            result['out_mean_phot_data'] = self.__out_mean_phot_data
        return result


ConfigManager.addHandler(ReferenceCatalogConfig)