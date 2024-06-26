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

from ElementsKernel.Logging import getLogger
from nnpz.config import ConfigManager
from nnpz.config.output import OutputHandlerConfig
from nnpz.config.reference import ReferenceConfig
from nnpz.config.target import TargetCatalogConfig
from nnpz.io.output_column_providers.UniformPhotometry import UniformPhotometry

logger = getLogger(__name__)


class UniformPhotometryConfig(ConfigManager.ConfigHandler):
    """
    Configure the output of the uniform photometry.

    See Also:
        UniformPhotometry for a detailed description of what it is and how it is computed.
    """

    def __init__(self):
        self.__added = False

    def __add_column_providers(self, args: Dict[str, Any]):
        corrected_phot = args.get('corrected_photometry', None)
        if not corrected_phot:
            return

        # Get dependencies
        target_config = ConfigManager.get_handler(TargetCatalogConfig).parse_args(args)
        ref_config = ConfigManager.get_handler(ReferenceConfig).parse_args(args)
        output_config = ConfigManager.get_handler(OutputHandlerConfig).parse_args(args)

        target_phot = target_config['target_photometry']
        ref_phot = ref_config['reference_photometry']

        # Drop output not possible to compute if a subset if filtered out
        filtered_corrected_phot = {}
        for target_cols, (a, b, target_a, target_b) in corrected_phot.items():
            if a in ref_phot.system.bands and b in target_phot.system.bands:
                filtered_corrected_phot[target_cols] = (a, b, target_a, target_b)
            else:
                logger.info('Drop corrected photometry for pair %s - %s', a, b)

        corrected_phot = filtered_corrected_phot

        # Build the list of tuples
        output_config['output_handler'].add_column_provider(
            UniformPhotometry(target_phot, ref_phot, corrected_phot)
        )

    def parse_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.__added:
            self.__add_column_providers(args)
            self.__added = True
        return {}


ConfigManager.add_handler(UniformPhotometryConfig)
