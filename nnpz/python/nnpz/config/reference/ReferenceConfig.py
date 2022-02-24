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
Created on: 06/04/18
Author: Nikolaos Apostolakos
"""
from typing import Any, Dict

import numpy as np
from nnpz.config import ConfigManager
from nnpz.config.reference.ReferenceSampleConfig import ReferenceSampleConfig
from nnpz.config.reference.ReferenceSamplePhotometryConfig import ReferenceSamplePhotometryConfig
from nnpz.exceptions import CorruptedFileException


class ReferenceConfig(ConfigManager.ConfigHandler):
    """
    Wraps the configuration of the reference sample and catalog, so they can be used
    interchangeably
    """

    def __init__(self):
        self.__validated = False

    def __validate(self, options: Dict[str, Any]):
        ref_sample_ids = options['reference_ids']
        ref_photo = options['reference_photometry']
        if not np.array_equal(ref_sample_ids, ref_photo.ids):
            raise CorruptedFileException('Reference sample IDs and photometry IDs do not match')

    def parse_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        options = {}
        options.update(ConfigManager.get_handler(ReferenceSampleConfig).parse_args(args))
        options.update(ConfigManager.get_handler(ReferenceSamplePhotometryConfig).parse_args(args))
        if not self.__validated:
            self.__validate(options)
        return options


ConfigManager.add_handler(ReferenceConfig)
