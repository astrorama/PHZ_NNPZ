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

from ElementsKernel import Logging
from nnpz.config import ConfigManager
from nnpz.reference_sample.ReferenceSample import ReferenceSample

logger = Logging.getLogger('Configuration')


class ReferenceSampleConfig(ConfigManager.ConfigHandler):
    """
    Load the reference sample, minus photometry

    See Also:
        ReferenceSamplePhotometryConfig
    """

    def __init__(self):
        self.__sample = None

    def __load_reference_sample(self, args: Dict[str, Any]):
        self._exists_parameter('reference_sample_dir', args)
        sample_dir = args['reference_sample_dir']
        logger.info('Reading reference sample from %s...', sample_dir)
        self.__sample = ReferenceSample(
            sample_dir,
            providers=args.get('reference_sample_providers', None)
        )
        logger.info('Reading reference sample done')

    def parse_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if self.__sample is None:
            self.__load_reference_sample(args)
        return {'reference_sample': self.__sample,
                'reference_ids': self.__sample.get_ids()}


ConfigManager.add_handler(ReferenceSampleConfig)
