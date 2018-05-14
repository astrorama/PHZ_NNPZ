"""
Created on: 06/04/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from nnpz.config import (ConfigManager, OutputHandlerConfig, TargetCatalogConfig,
                         ReferenceConfig)
import nnpz.io.output_column_providers as ocp


class MeanPhotOutputConfig(ConfigManager.ConfigHandler):

    def __init__(self):
        self.__added = False

    def __addColumnProvider(self, args):
        if args.get('neighbor_info_output', False):
            ref_options = ConfigManager.getHandler(ReferenceConfig).parseArgs(args)
            if 'out_mean_phot_filters' in ref_options:
                target_ids = ConfigManager.getHandler(TargetCatalogConfig).parseArgs(args)['target_ids']
                out_mean_phot_filters = ref_options['out_mean_phot_filters']
                out_mean_phot_data = ref_options['out_mean_phot_data']
                output = ConfigManager.getHandler(OutputHandlerConfig).parseArgs(args)['output_handler']
                output.addColumnProvider(ocp.MeanPhotometry(len(target_ids), out_mean_phot_filters, out_mean_phot_data))

    def parseArgs(self, args):
        if not self.__added:
            self.__addColumnProvider(args)
            self.__added = True
        return {}


ConfigManager.addHandler(MeanPhotOutputConfig)