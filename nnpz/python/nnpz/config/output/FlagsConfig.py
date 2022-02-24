#
# Copyright (C) 2012-2022 Euclid Science Ground Segment
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
from typing import Any, Dict

import nnpz.io.output_column_providers as ocp
from nnpz.config import ConfigManager
from nnpz.config.output import OutputHandlerConfig


class FlagsConfig(ConfigManager.ConfigHandler):
    """
    Configure the format of the flag columns: one per flag, or a single one with a bit per flag
    """

    def __init__(self):
        self.__added = False

    def __add_column_provider(self, args: Dict[str, Any]):
        separate_columns = False
        if 'flags_in_separate_columns' in args:
            separate_columns = args['flags_in_separate_columns']
        output = ConfigManager.get_handler(OutputHandlerConfig).parse_args(args)['output_handler']
        output.add_column_provider(ocp.Flags(separate_columns))

    def parse_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.__added:
            self.__add_column_provider(args)
            self.__added = True
        return {}


ConfigManager.add_handler(FlagsConfig)
