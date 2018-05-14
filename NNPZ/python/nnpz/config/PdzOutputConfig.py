"""
Created on: 06/04/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import nnpz.io.output_column_providers as ocp
from nnpz.config import (ConfigManager, OutputHandlerConfig, TargetCatalogConfig,
                         ReferenceConfig)
from nnpz.io.output_hdul_providers.PdzBins import PdzBins


class PdzOutputConfig(ConfigManager.ConfigHandler):

    def __init__(self):
        self.__added = False

    def __addColumnProvider(self, args):
        target_ids = ConfigManager.getHandler(TargetCatalogConfig).parseArgs(args)['target_ids']
        ref_options = ConfigManager.getHandler(ReferenceConfig).parseArgs(args)
        ref_ids = ref_options['reference_ids']

        # First handle the case where we have a reference sample directory. In
        # this case the PDZ is the weighted co-add of the sample PDZs.
        if 'reference_sample' in ref_options:
            ref_sample = ref_options['reference_sample']
            pdz_col = 'CoaddedPdz'
            pdz_prov = ocp.CoaddedPdz(len(target_ids), ref_sample, ref_ids)

        # Now we handle the case where we have a reference catalog. In this case
        # the PDZ is the normalized histogram of the neighbors redshifts.
        if 'reference_redshift' in ref_options:
            ref_z = ref_options['reference_redshift']
            pdz_col = 'TrueRedshiftPDZ'
            pdz_prov = ocp.TrueRedshiftPdz(len(target_ids), ref_z, 0, 6, 601)

        output = ConfigManager.getHandler(OutputHandlerConfig).parseArgs(args)['output_handler']
        output.addColumnProvider(pdz_prov)

        pdz_quantiles = args.get('pdz_quantiles', [])
        pdz_mc_samples = args.get('pdz_mc_samples', 0)
        output.addColumnProvider(ocp.PdfSampling(pdz_prov, pdz_col, quantiles=pdz_quantiles, mc_samples=pdz_mc_samples))

        output.addExtensionTableProvider(PdzBins(pdz_prov))

    def parseArgs(self, args):
        if not self.__added:
            self.__addColumnProvider(args)
            self.__added = True
        return {}


ConfigManager.addHandler(PdzOutputConfig)