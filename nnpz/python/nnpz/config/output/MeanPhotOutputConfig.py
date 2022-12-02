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
Created on: 06/04/18
Author: Nikolaos Apostolakos
"""
from typing import Any, Dict

import nnpz.io.output_column_providers as ocp
from ElementsKernel import Logging
from nnpz.config import ConfigManager
from nnpz.config.output import OutputHandlerConfig
from nnpz.config.reference import ReferenceConfig
from nnpz.config.target.TargetCatalogConfig import TargetCatalogConfig
from nnpz.photometry.photometry import Photometry
from nnpz.photometry.projection.source_independent_ebv import SourceIndependentGalacticEBV

logger = Logging.getLogger('Configuration')


class MeanPhotOutputConfig(ConfigManager.ConfigHandler):
    """
    Configure the output columns where to store the (weighted) mean photometry
    """

    def __init__(self):
        self.__added = False

    def __add_column_provider(self, args: Dict[str, Any]):
        ref_options = ConfigManager.get_handler(ReferenceConfig).parse_args(args)
        target_options = ConfigManager.get_handler(TargetCatalogConfig).parse_args(args)
        output_options = ConfigManager.get_handler(OutputHandlerConfig).parse_args(args)

        ref_phot: Photometry = ref_options['reference_photometry']

        if args.get('reference_sample_out_mean_phot_filters', None):
            out_mean_phot_filters = args['reference_sample_out_mean_phot_filters']
            out_mean_phot_idxs = ref_phot.system.get_band_indexes(out_mean_phot_filters)
            output = output_options['output_handler']

            if args.get('redden_mean_phot', False):
                self._exists_parameter('target_catalog_gal_ebv', args)

            output.add_column_provider(
                ocp.MeanPhotometry(out_mean_phot_filters, out_mean_phot_idxs, unit=ref_phot.unit)
            )

    def parse_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.__added:
            self.__add_column_provider(args)
            self.__added = True
        return {}


ConfigManager.add_handler(MeanPhotOutputConfig)
