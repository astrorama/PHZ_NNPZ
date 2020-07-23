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
Created on: 28/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import sys

from ElementsKernel import Logging
from nnpz.config import ConfigManager

logger = Logging.getLogger('Configuration')


class ReferenceUniqueConfig(ConfigManager.ConfigHandler):
    """
    This class only validates that only one of the reference catalog or sample is configured
    """

    def parseArgs(self, args):
        if 'reference_sample_dir' in args and 'reference_catalog' in args:
            logger.error('Only reference_sample_dir *or* reference_catalog options can be set')
            sys.exit(-1)
        if 'reference_sample_dir' not in args and 'reference_catalog' not in args:
            logger.error('One of reference_sample_dir and reference_catalog options must be set')
            sys.exit(-1)
        return {}


ConfigManager.addHandler(ReferenceUniqueConfig)
