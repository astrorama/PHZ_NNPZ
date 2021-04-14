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

import nnpz.io.output_column_providers as ocp
from ElementsKernel import Logging
from nnpz.config import ConfigManager
from nnpz.config.nnpz import NeighborListOutputConfig, OutputHandlerConfig, TargetCatalogConfig
from nnpz.config.reference import ReferenceConfig
from nnpz.io.output_hdul_providers.PdzBins import PdzBins

logger = Logging.getLogger('Configuration')


class PdzOutputConfig(ConfigManager.ConfigHandler):

    def __init__(self):
        self.__added = False
        self.__add_pdz_output = True

    def __addColumnProvider(self, args):
        target_ids = ConfigManager.getHandler(TargetCatalogConfig).parseArgs(args)['target_ids']
        ref_options = ConfigManager.getHandler(ReferenceConfig).parseArgs(args)
        neighbor_options = ConfigManager.getHandler(NeighborListOutputConfig).parseArgs(args)
        ref_ids = ref_options['reference_ids']

        # NNPZ can be run on neighbor-only mode, so skip PDZ computation in that case
        # Make sure the list of neighbors is generated in that case!
        self.__add_pdz_output = args.get('pdz_output', True)
        if not self.__add_pdz_output and not neighbor_options['neighbor_info_output']:
            logger.warning('PDZ and neighbor list disabled at the same time!')
            return

        # First handle the case where we have a reference sample directory. In
        # this case the PDZ is the weighted co-add of the sample PDZs.
        if 'reference_sample' in ref_options:
            ref_sample = ref_options['reference_sample']
            pdz_prov = ocp.CoaddedPdz(len(target_ids), ref_sample, ref_ids)

        # Now we handle the case where we have a reference catalog. In this case
        # the PDZ is the normalized histogram of the neighbors redshifts.
        if 'reference_redshift' in ref_options:
            ref_z = ref_options['reference_redshift']
            pdz_prov = ocp.TrueRedshiftPdz(len(target_ids), ref_z, 0, 6, 601)

        output = ConfigManager.getHandler(OutputHandlerConfig).parseArgs(args)['output_handler']
        output.addColumnProvider(pdz_prov)

        pdz_quantiles = args.get('pdz_quantiles', [])
        pdz_mc_samples = args.get('pdz_mc_samples', 0)
        if pdz_quantiles or pdz_mc_samples:
            pdf_sampling = ocp.PdfSampling(pdz_prov, quantiles=pdz_quantiles,
                                           mc_samples=pdz_mc_samples)
            output.addColumnProvider(pdf_sampling)
            output.addHeaderProvider(pdf_sampling)

        # Add point estimates
        if 'pdz_point_estimates' in args:
            output.addColumnProvider(ocp.PdzPointEstimates(pdz_prov, args['pdz_point_estimates']))

        output.addExtensionTableProvider(PdzBins(pdz_prov))

    def parseArgs(self, args):
        if not self.__added:
            self.__addColumnProvider(args)
            self.__added = True
        return {}


ConfigManager.addHandler(PdzOutputConfig)
