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
Created on: 06/04/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from ElementsKernel import Logging
import nnpz.io.output_column_providers as ocp
from nnpz.config import ConfigManager
from nnpz.config.nnpz import OutputHandlerConfig, TargetCatalogConfig, GalacticUnreddenerConfig
from nnpz.config.reference import ReferenceConfig
from nnpz.photometry.SourceIndependantGalacticUnReddening import \
    SourceIndependantGalacticUnReddening

logger = Logging.getLogger('Configuration')


class MeanPhotOutputConfig(ConfigManager.ConfigHandler):
    """
    Configure the output columns where to store the (weighted) mean photometry
    """

    def __init__(self):
        self.__added = False

    def __addColumnProvider(self, args):
        ref_options = ConfigManager.getHandler(ReferenceConfig).parseArgs(args)
        target_options = ConfigManager.getHandler(TargetCatalogConfig).parseArgs(args)
        output_options = ConfigManager.getHandler(OutputHandlerConfig).parseArgs(args)

        ref_phot_prov = ref_options['reference_photometry']

        if 'reference_sample_out_mean_phot_filters' in args:
            out_mean_phot_filters = args['reference_sample_out_mean_phot_filters']
            out_mean_phot_data = ref_phot_prov.getData(*out_mean_phot_filters)

            target_ids = target_options['target_ids']
            output = output_options['output_handler']

            target_ebv = None
            reddener = None

            if args.get('redden_mean_phot', False):
                self._checkParameterExists('target_catalog_gal_ebv', args)
                target_ebv = target_options['target_ebv']

                out_trans = dict()
                for f_name in out_mean_phot_filters:
                    out_trans[f_name] = ref_phot_prov.getFilterTransmission(f_name)

                reddener = SourceIndependantGalacticUnReddening(
                    out_trans, out_mean_phot_filters
                )

            output.addColumnProvider(
                ocp.MeanPhotometry(
                    len(target_ids), out_mean_phot_filters, out_mean_phot_data,
                    reddener, target_ebv
                )
            )

    def parseArgs(self, args):
        if not self.__added:
            self.__addColumnProvider(args)
            self.__added = True
        return {}


ConfigManager.addHandler(MeanPhotOutputConfig)
