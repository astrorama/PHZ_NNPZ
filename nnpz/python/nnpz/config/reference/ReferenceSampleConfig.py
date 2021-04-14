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
Created on: 28/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from ElementsKernel import Logging
from nnpz import ReferenceSample
from nnpz.config import ConfigManager

logger = Logging.getLogger('Configuration')


class ReferenceSampleConfig(ConfigManager.ConfigHandler):

    def __init__(self):
        self.__sample = None

    def __createSample(self, args):
        if 'reference_sample_dir' in args:
            sample_dir = args['reference_sample_dir']
            logger.info('Reading reference sample from %s...', sample_dir)
            self.__sample = ReferenceSample(
                sample_dir,
                providers=args.get('reference_sample_providers', None)
            )
            logger.info('Reading reference sample done')

    def parseArgs(self, args):
        if self.__sample is None:
            self.__createSample(args)

        if self.__sample is not None:
            return {'reference_sample': self.__sample,
                    'reference_ids': self.__sample.getIds()}
        return {}


ConfigManager.addHandler(ReferenceSampleConfig)
