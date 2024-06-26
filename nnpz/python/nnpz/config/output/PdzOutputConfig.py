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
from nnpz.config.output import NeighborListOutputConfig, OutputHandlerConfig
from nnpz.config.reference import ReferenceConfig
from nnpz.io.output_hdul_providers.PdzBins import PdzBins

logger = Logging.getLogger('Configuration')


class PdzOutputConfig(ConfigManager.ConfigHandler):
    """
    Configure the output of the co-added PDZ, quantiles and/or samples
    """

    def __init__(self):
        self.__added = False
        self.__add_pdz_output = True

    def __add_column_provider(self, args: Dict[str, Any]):
        ref_options = ConfigManager.get_handler(ReferenceConfig).parse_args(args)
        neighbor_options = ConfigManager.get_handler(NeighborListOutputConfig).parse_args(args)

        # NNPZ can be run on neighbor-only mode, so skip PDZ computation in that case
        # Make sure the list of neighbors is generated in that case!
        self.__add_pdz_output = args.get('pdz_output', True)
        if not self.__add_pdz_output and not neighbor_options['neighbor_info_output']:
            logger.warning('PDZ and neighbor list disabled at the same time!')
            return

        if not self.__add_pdz_output:
            return

        ref_sample = ref_options['reference_sample']

        kernel = args.get('pdz_kernel', None)
        kernel_bandwidth = args.get('pdz_kernel_bandwidth', 'auto')

        output = ConfigManager.get_handler(OutputHandlerConfig).parse_args(args)['output_handler']
        output.add_column_provider(
            ocp.CoaddedPdz(ref_sample, kernel=kernel, bandwidth=kernel_bandwidth,
                           store_kernel=args.get('pdz_kernel_store', False))
        )

        pdz_quantiles = args.get('pdz_quantiles', [])
        pdz_mc_samples = args.get('pdz_mc_samples', 0)
        if pdz_quantiles or pdz_mc_samples:
            pdf_sampling = ocp.PdfSampling(ref_sample, quantiles=pdz_quantiles,
                                           mc_samples=pdz_mc_samples)
            output.add_column_provider(pdf_sampling)
            output.add_header_provider(pdf_sampling)

        # Add point estimates
        if 'pdz_point_estimates' in args:
            output.add_column_provider(
                ocp.PdzPointEstimates(ref_sample, args['pdz_point_estimates']))

        output.add_extension_table_provider(PdzBins(ref_sample))

    def parse_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.__added:
            self.__add_column_provider(args)
            self.__added = True
        return {}


ConfigManager.add_handler(PdzOutputConfig)
