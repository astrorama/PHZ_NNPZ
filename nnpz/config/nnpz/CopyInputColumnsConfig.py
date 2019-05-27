"""
Created on: 28/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from nnpz.config import ConfigManager
from nnpz.config.nnpz import (TargetCatalogConfig, OutputHandlerConfig)
import nnpz.io.output_column_providers as ocp


class CopyInputColumnsConfig(ConfigManager.ConfigHandler):

    def __init__(self):
        self.__added = False

    def __addColumnProvider(self, args):
        target_config = ConfigManager.getHandler(TargetCatalogConfig).parseArgs(args)
        cat_to_copy = target_config['target_astropy_table']
        output = ConfigManager.getHandler(OutputHandlerConfig).parseArgs(args)['output_handler']

        do_copy = args.get('copy_input_columns', False)
        if do_copy:
            output.addColumnProvider(ocp.CatalogCopy(cat_to_copy))
        else:
            id_column = target_config['target_id_column']
            output.addColumnProvider(ocp.CatalogCopy(cat_to_copy, [id_column]))


    def parseArgs(self, args):
        if not self.__added:
            self.__addColumnProvider(args)
            self.__added = True
        return {}


ConfigManager.addHandler(CopyInputColumnsConfig)