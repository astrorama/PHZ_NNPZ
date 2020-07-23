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
Created on: 23/04/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from nnpz import NnpzFlag
from nnpz.config import ConfigManager
from nnpz.config.nnpz import (TargetCatalogConfig, OutputHandlerConfig)
import nnpz.io.output_column_providers as ocp


class FlagsConfig(ConfigManager.ConfigHandler):
    """
    Configure the format of the flag columns: one per flag, or a single one with a bit per flag
    """

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
