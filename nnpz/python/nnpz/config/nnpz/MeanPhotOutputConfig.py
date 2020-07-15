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
Created on: 06/04/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import nnpz.io.output_column_providers as ocp
from ElementsKernel import Logging
from nnpz.config import ConfigManager
from nnpz.config.nnpz import OutputHandlerConfig, TargetCatalogConfig, GalacticUnreddenerConfig
from nnpz.config.reference import ReferenceConfig

logger = Logging.getLogger('Configuration')


class MeanPhotOutputConfig(ConfigManager.ConfigHandler):

    def __init__(self):
        self.__added = False
        self.__redden = False

    def __addColumnProvider(self, args):
        if args.get('neighbor_info_output', False):
            ref_options = ConfigManager.getHandler(ReferenceConfig).parseArgs(args)
            if 'out_mean_phot_filters' in ref_options:
                target_options = ConfigManager.getHandler(TargetCatalogConfig).parseArgs(args)
                self.__redden = args.get('redden_mean_phot', False)
                target_ids = target_options['target_ids']
                out_mean_phot_filters = ref_options['out_mean_phot_filters']
                out_mean_phot_data = ref_options['out_mean_phot_data']
                output = ConfigManager.getHandler(OutputHandlerConfig).parseArgs(args)['output_handler']

                if self.__redden:
                    unredden_config = ConfigManager.getHandler(GalacticUnreddenerConfig).parseArgs(args)
                    unreddener = unredden_config['galactic_absorption_unreddener']
                    if not unreddener:
                        logger.error('Galactic un-reddening must be configured for using the reddening on the output')
                    target_ebv = target_options['target_ebv']
                else:
                    unreddener = None
                    target_ebv = None

                output.addColumnProvider(
                    ocp.MeanPhotometry(
                        len(target_ids), out_mean_phot_filters, out_mean_phot_data, unreddener, target_ebv
                    )
                )

    def parseArgs(self, args):
        if not self.__added:
            self.__addColumnProvider(args)
            self.__added = True
        return {}


ConfigManager.addHandler(MeanPhotOutputConfig)
