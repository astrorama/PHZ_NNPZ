"""
Created on: 28/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from nnpz.config import (ConfigManager, TargetCatalogConfig, OutputHandlerConfig)
import nnpz.io.output_column_providers as ocp


class CopyInputColumnsConfig(ConfigManager.ConfigHandler):

    def __init__(self):
        self.__added = False

    def __addColumnProvider(self, args):
        do_copy = args.get('copy_input_columns', False)
        if do_copy:
            cat_to_copy = ConfigManager.getHandler(TargetCatalogConfig).parseArgs(args)['target_astropy_table']
            output = ConfigManager.getHandler(OutputHandlerConfig).parseArgs(args)['output_handler']
            output.addColumnProvider(ocp.FullCatalogCopy(cat_to_copy))

    def parseArgs(self, args):
        if not self.__added:
            self.__addColumnProvider(args)
            self.__added = True
        return {}


ConfigManager.addHandler(CopyInputColumnsConfig)