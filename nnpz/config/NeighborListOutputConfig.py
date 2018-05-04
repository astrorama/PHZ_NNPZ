"""
Created on: 06/04/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from nnpz.config import (ConfigManager, OutputHandlerConfig, TargetCatalogConfig,
                         ReferenceConfig)
import nnpz.io.output_column_providers as ocp


class NeighborListOutputConfig(ConfigManager.ConfigHandler):

    def __init__(self):
        self.__added = False


    def __addColumnProvider(self, args):
        if args.get('neighbor_info_output', False):
            target_ids = ConfigManager.getHandler(TargetCatalogConfig).parseArgs(args)['target_ids']
            ref_ids = ConfigManager.getHandler(ReferenceConfig).parseArgs(args)['reference_ids']
            output = ConfigManager.getHandler(OutputHandlerConfig).parseArgs(args)['output_handler']
            output.addColumnProvider(ocp.NeighborList(len(target_ids), ref_ids, args['neighbors_no']))

    def parseArgs(self, args):
        if not self.__added:
            self.__addColumnProvider(args)
            self.__added = True
        return {}


ConfigManager.addHandler(NeighborListOutputConfig)