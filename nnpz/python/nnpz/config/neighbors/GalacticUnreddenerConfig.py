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
Created on: 10/07/18
Author: Florian Dubath
"""

from ElementsKernel import Logging
from nnpz.config import ConfigManager
from nnpz.config.target import TargetCatalogConfig
from nnpz.photometry.projection.source_independent_ebv import SourceIndependentGalacticEBV

logger = Logging.getLogger('Configuration')


class GalacticUnreddenerConfig(ConfigManager.ConfigHandler):
    """
    Configure the un-reddening: remove the reddening effect of the galactic plane so
    the object photometry is more within the reference color space
    """

    def __init__(self):
        self.__galactic_absorption_unreddener = None
        self.__parsed = False

    def __createGalacticUnreddener(self, args):
        target_config = ConfigManager.getHandler(TargetCatalogConfig).parseArgs(args)
        target_photo = target_config['target_photometry']
        if 'ebv' in target_photo.colorspace:
            logger.info('Target catalog EBV present')
            self.__galactic_absorption_unreddener = SourceIndependentGalacticEBV(
                system=target_photo.system
            )
        self.__parsed = True

    def parseArgs(self, args):
        if not self.__parsed:
            self.__createGalacticUnreddener(args)
        return {
            'source_independent_ebv': self.__galactic_absorption_unreddener
        }


ConfigManager.addHandler(GalacticUnreddenerConfig)
