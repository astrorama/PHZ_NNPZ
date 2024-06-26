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
from typing import Any, Dict

from nnpz.config import ConfigManager


class NeighborsCatalog(ConfigManager.ConfigHandler):
    """
    Configure the intermediate catalog to use when running stages
    (neighbor find, photometry correction, weighting and output) separately.

    This has *no* effect when running the single `nnpz` command.
    """

    def __init__(self):
        self.__catalog = None

    def __setup(self, args: Dict[str, Any]):
        self.__catalog = args.get('neighbor_catalog', 'neighbors.fits')

    def parse_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if self.__catalog is None:
            self.__setup(args)
        return {
            'neighbor_catalog': self.__catalog,
        }


ConfigManager.add_handler(NeighborsCatalog)
