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
Created on: 28/02/18
Author: Nikolaos Apostolakos
"""
from typing import Any, Dict

from nnpz.config import ConfigManager
from nnpz.io import OutputHandler


class OutputHandlerConfig(ConfigManager.ConfigHandler):
    """
    Configure the destination for the output catalog
    """

    def __init__(self):
        self.__output_handler = None

    def add_column_provider(self, column_provider: OutputHandler.OutputColumnProviderInterface):
        self.__output_handler.add_column_provider(column_provider)

    def __set_output_handler(self, args: Dict[str, Any]):
        self._exists_parameter('output_file', args)
        self.__output_handler = OutputHandler(args['output_file'])

    def parse_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if self.__output_handler is None:
            self.__set_output_handler(args)
        return {'output_handler': self.__output_handler}


ConfigManager.add_handler(OutputHandlerConfig)
