"""
Created on: 28/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from nnpz.config import ConfigManager
from nnpz.io import OutputHandler

_output_handler = OutputHandler()

class OutputHandlerConfig(ConfigManager.ConfigHandler):

    @staticmethod
    def addColumnProvider(column_provider):
        _output_handler.addColumnProvider(column_provider)

    def parseArgs(self, args):
        self._checkParameterExists('output_file', args)
        return {'output_handler' : _output_handler,
                'output_file' : args['output_file']}


ConfigManager.addHandler(OutputHandlerConfig)