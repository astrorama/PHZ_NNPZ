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

import nnpz.io.output_column_providers as ocp
import numpy as np
from nnpz.config import ConfigManager
from nnpz.config.output import OutputHandlerConfig
from nnpz.config.target import TargetCatalogConfig


class CopyInputColumnsConfig(ConfigManager.ConfigHandler):
    """
    Copy into the output catalog columns from the input
    """

    def __init__(self):
        self.__added = False

    @staticmethod
    def __add_column_provider(args: Dict[str, Any]):
        target_config = ConfigManager.get_handler(TargetCatalogConfig).parse_args(args)
        cat_to_copy = target_config['target_hdu']
        output = ConfigManager.get_handler(OutputHandlerConfig).parse_args(args)['output_handler']

        do_copy = args.get('copy_input_columns', False)
        if do_copy:
            output.add_column_provider(ocp.CatalogCopy(cat_to_copy.get_rec_dtype()[0], cat_to_copy))
        else:
            id_column = target_config['target_id_column']
            output.add_column_provider(ocp.CatalogCopy(np.dtype([id_column]), cat_to_copy))

    def parse_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.__added:
            self.__add_column_provider(args)
            self.__added = True
        return {}


ConfigManager.add_handler(CopyInputColumnsConfig)
