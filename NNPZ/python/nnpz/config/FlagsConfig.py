"""
Created on: 23/04/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from nnpz import NnpzFlag
from nnpz.config import (ConfigManager, TargetCatalogConfig, OutputHandlerConfig)
import nnpz.io.output_column_providers as ocp


class FlagsConfig(ConfigManager.ConfigHandler):

    def __init__(self):
        self.__flag_list = None

    def __createFlagList(self, args):
        target_size = ConfigManager.getHandler(TargetCatalogConfig).parseArgs(args)['target_size']
        self.__flag_list = [NnpzFlag() for i in range(target_size)]

    def __addColumnProvider(self, args):
        separate_columns = False
        if 'flags_in_separate_columns' in args:
            separate_columns = args['flags_in_separate_columns']
        output = ConfigManager.getHandler(OutputHandlerConfig).parseArgs(args)['output_handler']
        output.addColumnProvider(ocp.Flags(self.__flag_list, separate_columns))

    def parseArgs(self, args):
        if self.__flag_list is None:
            self.__createFlagList(args)
            self.__addColumnProvider(args)
        return {'flag_list' : self.__flag_list}

ConfigManager.addHandler(FlagsConfig)