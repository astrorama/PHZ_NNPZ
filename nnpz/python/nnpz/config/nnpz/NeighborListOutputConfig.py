#
# Copyright (C) 2012-2021 Euclid Science Ground Segment
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
Created on: 06/04/18
Author: Nikolaos Apostolakos
"""


import nnpz.io.output_column_providers as ocp
from nnpz.config import ConfigManager
from nnpz.config.nnpz import NeighborSelectorConfig, OutputHandlerConfig
from nnpz.config.reference import ReferenceConfig


class NeighborListOutputConfig(ConfigManager.ConfigHandler):
    """
    Enable or disable neighbor information on the output catalog
    """

    def __init__(self):
        self.__added = False
        self.__neighbor_info_output = False

    def __addColumnProvider(self, args):
        self.__neighbor_info_output = args.get('neighbor_info_output', False)
        if self.__neighbor_info_output:
            ref_ids = ConfigManager.getHandler(ReferenceConfig).parseArgs(args)['reference_ids']
            output = ConfigManager.getHandler(OutputHandlerConfig).parseArgs(args)['output_handler']
            nn = ConfigManager.getHandler(NeighborSelectorConfig).parseArgs(args)['neighbor_no']
            output.addColumnProvider(ocp.NeighborList(ref_ids, n_neighbors=nn))

    def parseArgs(self, args):
        if not self.__added:
            self.__addColumnProvider(args)
            self.__added = True
        return {
            'neighbor_info_output': self.__neighbor_info_output
        }


ConfigManager.addHandler(NeighborListOutputConfig)
