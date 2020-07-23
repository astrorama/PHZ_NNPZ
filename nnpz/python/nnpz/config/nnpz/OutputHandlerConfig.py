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

from nnpz.config import ConfigManager
from nnpz.io import OutputHandler

_output_handler = OutputHandler()


class OutputHandlerConfig(ConfigManager.ConfigHandler):
    """
    Configure the destination for the output catalog
    """

    @staticmethod
    def addColumnProvider(column_provider):
        _output_handler.addColumnProvider(column_provider)

    def parseArgs(self, args):
        self._checkParameterExists('output_file', args)
        return {'output_handler': _output_handler,
                'output_file': args['output_file']}


ConfigManager.addHandler(OutputHandlerConfig)
