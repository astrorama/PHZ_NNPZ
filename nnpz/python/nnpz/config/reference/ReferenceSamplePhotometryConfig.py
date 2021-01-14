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

import numpy as np
from ElementsKernel import Logging
from nnpz.config import ConfigManager
from nnpz.config.reference import ReferenceSampleConfig
from nnpz.reference_sample import PhotometryProvider

logger = Logging.getLogger('Configuration')


class ReferenceSamplePhotometryConfig(ConfigManager.ConfigHandler):
    """
    Configure the table containing the photometry of the reference objects, and
    the bands to use for the search
    """

    def __init__(self):
        self.__ref_phot_data = None
        self.__ref_phot_prov = None
        self.__ref_filters = None
        self.__out_mean_phot_filters = None
        self.__out_mean_phot_data = None
        self.__phot_file = None
        self.__ref_phot_type = None

    def __createData(self, args):

        ref_sample_dict = ConfigManager.getHandler(ReferenceSampleConfig).parseArgs(args)
        if 'reference_sample' in ref_sample_dict:

            self._checkParameterExists('reference_sample_phot_file', args)
            self.__phot_file = args['reference_sample_phot_file']
            self._checkParameterExists('reference_sample_phot_filters', args)
            phot_filters = args['reference_sample_phot_filters']

            logger.info('Using reference sample photometry from %s', self.__phot_file)
            self.__ref_phot_prov = PhotometryProvider(self.__phot_file)
            if np.any(self.__ref_phot_prov.getIds() != ref_sample_dict['reference_ids']):
                logger.error('ERROR: Reference sample photometry ID mismatch')
                exit(-1)

            logger.info('Reference sample photometric bands: %s', phot_filters)
            self.__ref_phot_data = self.__ref_phot_prov.getData(*phot_filters)
            self.__ref_phot_type = self.__ref_phot_prov.getType()
            logger.info('Reference sample photometry type: %s', self.__ref_phot_type)
            self.__ref_filters = {}
            for f_name in phot_filters:
                self.__ref_filters[f_name] = self.__ref_phot_prov.getFilterTransmission(f_name)

    def parseArgs(self, args):

        if self.__ref_phot_data is None and self.__out_mean_phot_data is None:
            self.__createData(args)

        result = {}
        if self.__ref_phot_data is not None:
            result['reference_sample_phot_file'] = self.__phot_file
            result['reference_phot_type'] = self.__ref_phot_type
            result['reference_photometry'] = self.__ref_phot_prov
            result['reference_phot_data'] = self.__ref_phot_data
            result['reference_filter_transmission'] = self.__ref_filters
        return result


ConfigManager.addHandler(ReferenceSamplePhotometryConfig)
