"""
Created on: 06/04/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from nnpz.config import (ConfigManager, OutputHandlerConfig, TargetCatalogConfig,
                         ReferenceConfig)
import nnpz.io.output_column_providers as ocp


class PdzOutputConfig(ConfigManager.ConfigHandler):

    def __init__(self):
        self.__pdz_column_prov = None

    def __addColumnProvider(self, args):
        target_ids = ConfigManager.getHandler(TargetCatalogConfig).parseArgs(args)['target_ids']
        ref_options = ConfigManager.getHandler(ReferenceConfig).parseArgs(args)
        ref_ids = ref_options['reference_ids']

        # First handle the case where we have a reference sample directory. In
        # this case the PDZ is the weighted co-add of the sample PDZs.
        if 'reference_sample' in ref_options:
            ref_sample = ref_options['reference_sample']
            self.__pdz_column_prov = ocp.CoaddedPdz(len(target_ids), ref_sample, ref_ids)

        # Now we handle the case where we have a reference catalog. In this case
        # the PDZ is the normalized histogram of the neighbors redshifts.
        if 'reference_redshift' in ref_options:
            ref_z = ref_options['reference_redshift']
            self.__pdz_column_prov = ocp.TrueRedshiftPdz(len(target_ids), ref_z, 0, 6, 601)

        output = ConfigManager.getHandler(OutputHandlerConfig).parseArgs(args)['output_handler']
        output.addColumnProvider(self.__pdz_column_prov)


    def parseArgs(self, args):
        if self.__pdz_column_prov is None:
            self.__addColumnProvider(args)
        return {'pdz_column_provider' : self.__pdz_column_prov}


ConfigManager.addHandler(PdzOutputConfig)