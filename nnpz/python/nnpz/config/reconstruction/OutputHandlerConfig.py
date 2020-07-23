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
Created on: 27/05/19
Author: Alejandro Alvarez Ayllon
"""

from __future__ import division, print_function

import nnpz.io.output_column_providers as ocp
from nnpz.config import ConfigManager
from nnpz.config.reconstruction import NnpzCatalogConfig
from nnpz.config.reference import ReferenceConfig
from nnpz.io import OutputHandler
from nnpz.io.output_hdul_providers.PdzBins import PdzBins

_output_handler = OutputHandler()


class OutputHandlerConfig(ConfigManager.ConfigHandler):
    """
    Configuration of the output properties to enable
    """

    def __init__(self):
        self.__added = False

    @staticmethod
    def addColumnProvider(column_provider):
        """
        Register a new column provider
        Args:
            column_provider: OutputHandler.OutputColumnProviderInterface
        """
        _output_handler.addColumnProvider(column_provider)

    @staticmethod
    def _addColumnProviders(args):
        nnpz_config = ConfigManager.getHandler(NnpzCatalogConfig).parseArgs(args)
        ref_config = ConfigManager.getHandler(ReferenceConfig).parseArgs(args)
        nnpz_cat = nnpz_config['nnpz_astropy_table']
        nnpz_size = nnpz_config['nnpz_size']
        ref_ids = ref_config['reference_ids']

        # Copy NNPZ catalog as-is
        _output_handler.addColumnProvider(ocp.CatalogCopy(nnpz_cat))

        # First handle the case where we have a reference sample directory. In
        # this case the PDZ is the weighted co-add of the sample PDZs.
        if 'reference_sample' in ref_config:
            ref_sample = ref_config['reference_sample']
            pdz_prov = ocp.CoaddedPdz(nnpz_size, ref_sample, ref_ids)

        # Now we handle the case where we have a reference catalog. In this case
        # the PDZ is the normalized histogram of the neighbors redshifts.
        if 'reference_redshift' in ref_config:
            ref_z = ref_config['reference_redshift']
            pdz_prov = ocp.TrueRedshiftPdz(nnpz_size, ref_z, 0, 6, 601)

        _output_handler.addColumnProvider(pdz_prov)

        pdz_quantiles = args.get('pdz_quantiles', [])
        pdz_mc_samples = args.get('pdz_mc_samples', 0)
        pdf_sampling = ocp.PdfSampling(pdz_prov, quantiles=pdz_quantiles, mc_samples=pdz_mc_samples)
        _output_handler.addColumnProvider(pdf_sampling)
        _output_handler.addHeaderProvider(pdf_sampling)

        # Add point estimates
        if 'pdz_point_estimates' in args:
            _output_handler.addColumnProvider(
                ocp.PdzPointEstimates(pdz_prov, args['pdz_point_estimates'])
            )

        _output_handler.addExtensionTableProvider(PdzBins(pdz_prov))

    def parseArgs(self, args):
        self._checkParameterExists('pdz_output_file', args)
        if not self.__added:
            self._addColumnProviders(args)
            self.__added = True
        return {
            'pdz_output_file': args['pdz_output_file'],
            'pdz_output_handler': _output_handler
        }


ConfigManager.addHandler(OutputHandlerConfig)
