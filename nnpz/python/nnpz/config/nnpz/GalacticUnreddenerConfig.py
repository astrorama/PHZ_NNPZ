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
Created on: 10/07/18
Author: Florian Dubath
"""

from __future__ import division, print_function

from ElementsKernel import Logging
from nnpz.config import ConfigManager
from nnpz.config.reference import ReferenceConfig
from nnpz.photometry.SourceIndependantGalacticUnReddening import \
    SourceIndependantGalacticUnReddening

logger = Logging.getLogger('Configuration')


class GalacticUnreddenerConfig(ConfigManager.ConfigHandler):
    """
    Configure the un-reddening: remove the reddening effect of the galactic plane so
    the object photometry is more within the reference color space
    """

    def __init__(self):
        self.__galactic_absorption_unreddener = None
        self.__apply = False
        self.__parsed = False

    def __createGalacticUnreddener(self, args):
        if 'target_catalog_gal_ebv' in args:
            self.__apply = True

            self._checkParameterExists('reference_sample_phot_filters', args)

            reference_config = ConfigManager.getHandler(ReferenceConfig).parseArgs(args)
            filter_order = args['reference_sample_phot_filters']
            filter_trans_map = reference_config['reference_filter_transmission']

            self.__galactic_absorption_unreddener = SourceIndependantGalacticUnReddening(
                filter_trans_map,
                filter_order
            )
        self.__parsed = True


    def parseArgs(self, args):
        if not self.__parsed:
            self.__createGalacticUnreddener(args)
        return {
            'apply_galactic_absorption': self.__apply,
            'galactic_absorption_unreddener': self.__galactic_absorption_unreddener
        }


ConfigManager.addHandler(GalacticUnreddenerConfig)
