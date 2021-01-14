#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
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

from __future__ import division, print_function

import nnpz.io.catalog_properties as prop
from ElementsKernel import Logging
from nnpz.config import ConfigManager
from nnpz.io import CatalogReader
from nnpz.reference_catalog.ReferenceCatalog import ReferenceCatalog
from nnpz.reference_catalog.ReferenceCatalogPhotometryProvider import CatalogPhotometryProvider

logger = Logging.getLogger('Configuration')


class ReferenceCatalogConfig(ConfigManager.ConfigHandler):
    """
    Configure a reference *catalog*: a table with the set of reference objects
    """

    def __init__(self):
        self.__ref_ids = None
        self.__ref_phot_data = None
        self.__ref_z = None
        self.__ref_z_bins = None
        self.__ref_catalog = None
        self.__ref_cat = None
        self.__ref_phot_prov = None

    def __createData(self, args):
        if 'reference_catalog' in args:
            self.__ref_cat = args['reference_catalog']
            self._checkParameterExists('reference_catalog_filters', args)
            ref_filters = args['reference_catalog_filters']
            self._checkParameterExists('reference_catalog_redshift', args)
            ref_z_col = args['reference_catalog_redshift']

            logger.info('Reading reference catalog from %s...', self.__ref_cat)
            ref_reader = CatalogReader(self.__ref_cat)
            self.__ref_phot_prov = CatalogPhotometryProvider(ref_reader)
            self.__ref_ids = ref_reader.get(prop.ID)
            self.__ref_phot_data = ref_reader.get(prop.Photometry(ref_filters))
            self.__ref_z = ref_reader.get(prop.Column(ref_z_col))

            # The redshift column is a PDZ instead of a point estimate
            if len(self.__ref_z.shape) > 1:
                self._checkParameterExists('reference_catalog_redshift_bins_hdu', args)
                bins_reader = CatalogReader(
                    self.__ref_cat, hdu=args['reference_catalog_redshift_bins_hdu']
                )
                self.__ref_z_bins = bins_reader.get(
                    prop.Column(args.get('reference_catalog_redshift_bins_col', 'BINS_PDF'))
                )
                self.__ref_catalog = ReferenceCatalog(self.__ref_ids, self.__ref_z,
                                                      self.__ref_z_bins)

    def parseArgs(self, args):

        if self.__ref_ids is None:
            self.__createData(args)

        result = {}
        if self.__ref_ids is not None:
            result['reference_catalog'] = self.__ref_cat
            result['reference_ids'] = self.__ref_ids
            result['reference_phot_data'] = self.__ref_phot_data
            result['reference_photometry'] = self.__ref_phot_prov
            if self.__ref_catalog:
                result['reference_sample'] = self.__ref_catalog
            else:
                result['reference_redshift'] = self.__ref_z
        return result


ConfigManager.addHandler(ReferenceCatalogConfig)
