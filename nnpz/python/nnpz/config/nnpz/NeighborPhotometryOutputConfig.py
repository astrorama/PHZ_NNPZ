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

import nnpz.io.output_column_providers as ocp
from ElementsKernel import Logging
from nnpz.config import ConfigManager
from nnpz.config.nnpz import NeighborSelectorConfig, OutputHandlerConfig
from nnpz.config.reference import ReferenceSamplePhotometryConfig

logger = Logging.getLogger('Configuration')


class NeighborPhotometryOutputConfig(ConfigManager.ConfigHandler):
    """
    Enable or disable *matched* neighbor photometry on the output catalog
    """

    def __init__(self):
        self.__added = False

    def __addColumnProvider(self, args):
        neighbor_photo_output = args.get('neighbor_photometry_output', False)
        if neighbor_photo_output:
            logger.warning(
                'Neighbor photometry output enabled! This is intended for debugging only!')
            output = ConfigManager.getHandler(OutputHandlerConfig).parseArgs(args)['output_handler']
            nn = ConfigManager.getHandler(NeighborSelectorConfig).parseArgs(args)['neighbor_no']
            filters = ConfigManager.getHandler(ReferenceSamplePhotometryConfig).parseArgs(args)[
                'reference_filters']
            output.addColumnProvider(ocp.NeighborPhotometry(filters, n_neighbors=nn))

    def parseArgs(self, args):
        if not self.__added:
            self.__addColumnProvider(args)
            self.__added = True
        return {}


ConfigManager.addHandler(NeighborPhotometryOutputConfig)
