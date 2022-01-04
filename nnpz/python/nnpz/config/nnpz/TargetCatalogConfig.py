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
Created on: 28/02/18
Author: Nikolaos Apostolakos
"""


import nnpz.io.catalog_properties as prop
from ElementsKernel import Logging
from nnpz.config import ConfigManager
from nnpz.io import CatalogReader

logger = Logging.getLogger('Configuration')


class TargetCatalogConfig(ConfigManager.ConfigHandler):
    """
    Configure the input catalog containing the target objects
    """

    def __init__(self):
        self.__target_size = None
        self.__target_id_column = None
        self.__target_ids = None
        self.__target_phot_data = None
        self.__target_astropy_table = None
        self.__target_ebv = None
        self.__target_filter_mean_wavelength = None
        self.__target_filters = None

    def __createData(self, args):

        self._checkParameterExists('target_catalog', args)
        target_cat = args['target_catalog']
        self._checkParameterExists('target_catalog_filters', args)
        missing_phot_flags = args.get('missing_photometry_flags', [])
        self.__target_id_column = args.get('target_catalog_id_column', 'ID')

        self.__target_filters = args['target_catalog_filters']
        logger.info('Target catalog photometric columns: %s', self.__target_filters)

        logger.info('Reading target catalog: %s', target_cat)
        target_reader = CatalogReader(target_cat)
        self.__target_ids = target_reader.get(prop.ID(self.__target_id_column))
        self.__target_phot_data = target_reader.get(
            prop.Photometry(self.__target_filters, missing_phot_flags)
        )
        self.__target_astropy_table = target_reader.getAsAstropyTable()

        target_catalog_ebv = args.get('target_catalog_gal_ebv', None)
        if target_catalog_ebv is not None:
            logger.info('Using E(B-V) columns %s', target_catalog_ebv)
            self.__target_ebv = target_reader.get(
                prop.EBV(target_catalog_ebv, nan_flags=missing_phot_flags)
            )

        target_catalog_filters_mean = args.get('target_catalog_filters_mean', None)
        if target_catalog_filters_mean is not None:
            logger.info('Using filters mean wavelength columns %s', target_catalog_filters_mean)
            self.__target_filter_mean_wavelength = target_reader.get(
                prop.FiltersMeanWavelength(target_catalog_filters_mean, missing_phot_flags + [0.])
            )

        if 'input_size' in args:
            input_size = args['input_size']
            if not isinstance(input_size, slice):
                input_size = slice(input_size)
            logger.warning('Processing only %s objects from target catalog', input_size)
            self.__target_ids = self.__target_ids[input_size]
            self.__target_phot_data = self.__target_phot_data[input_size]
            self.__target_astropy_table = self.__target_astropy_table[input_size]

        self.__target_size = len(self.__target_ids)
        self.__target_chunk_size = args.get('input_chunk_size', max(self.__target_size, 1))

    def parseArgs(self, args):

        if self.__target_ids is None:
            self.__createData(args)

        return {
            'target_id_column': self.__target_id_column,
            'target_ids': self.__target_ids,
            'target_phot_data': self.__target_phot_data,
            'target_filters': self.__target_filters,
            'target_ebv': self.__target_ebv,
            'target_filter_mean_wavelength': self.__target_filter_mean_wavelength,
            'target_astropy_table': self.__target_astropy_table,
            'target_size': self.__target_size,
            'target_chunk_size': self.__target_chunk_size
        }


ConfigManager.addHandler(TargetCatalogConfig)
