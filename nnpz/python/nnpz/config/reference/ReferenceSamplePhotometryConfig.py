#
# Copyright (C) 2012-2022 Euclid Science Ground Segment
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

from collections import OrderedDict
from typing import Any, Dict

from ElementsKernel import Logging
from astropy import units as u
from nnpz.config import ConfigManager
from nnpz.photometry.colorspace import RestFrame
from nnpz.photometry.photometric_system import PhotometricSystem
from nnpz.photometry.photometry import Photometry
from nnpz.reference_sample import PhotometryProvider

logger = Logging.getLogger('Configuration')


class ReferenceSamplePhotometryConfig(ConfigManager.ConfigHandler):
    """
    Configure the table containing the photometry of the reference objects, and
    the bands to use for the search
    """

    def __init__(self):
        self.__ref_photo = None
        self.__ebv_corr = None
        self.__filter_corr = None

    def __load_photometry(self, args: Dict[str, Any]):
        self._exists_parameter('reference_sample_phot_file', args)

        file = args['reference_sample_phot_file']

        logger.info('Using reference sample photometry from %s', file)
        ref_phot_prov = PhotometryProvider(file)
        ref_filters = ref_phot_prov.get_filter_list()

        logger.info('Reference sample %d photometric bands: %s', len(ref_filters), ref_filters)

        ref_type = ref_phot_prov.get_type()
        if ref_type != 'F_nu_uJy':
            raise ValueError(f'Only F_nu_uJy accepted as reference photometry type, got {ref_type}')
        logger.info('Reference sample photometry type: %s', ref_type)
        ref_data = ref_phot_prov.get_data(ref_filters)

        filter_trans = OrderedDict()
        for f_name in ref_filters:
            filter_trans[f_name] = ref_phot_prov.get_filter_transmission(f_name)

        self.__ref_photo = Photometry(ref_phot_prov.get_ids(),
                                      values=u.Quantity(ref_data, u.uJy, copy=False),
                                      system=PhotometricSystem(filter_trans), colorspace=RestFrame)
        self.__ebv_corr = ref_phot_prov.get_ebv_correction_factors(ref_filters)
        self.__filter_corr = ref_phot_prov.get_shift_correction_factors(ref_filters)

    def parse_args(self, args: Dict[str, Any]):
        if self.__ref_photo is None:
            self.__load_photometry(args)
        return {
            'reference_photometry': self.__ref_photo,
            'reference_ebv_correction': self.__ebv_corr,
            'reference_filter_variation_correction': self.__filter_corr,
            'reference_system': self.__ref_photo.system,
        }


ConfigManager.add_handler(ReferenceSamplePhotometryConfig)
