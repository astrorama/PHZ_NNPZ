"""
Created on: 06/04/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from nnpz.config import ConfigManager
from nnpz.config.nnpz import (ReferenceSampleConfig, ReferenceCatalogConfig,
                              ReferenceSamplePhotometryConfig)


class ReferenceConfig(ConfigManager.ConfigHandler):

    def parseArgs(self, args):
        options = {}
        options.update(ConfigManager.getHandler(ReferenceSampleConfig).parseArgs(args))
        options.update(ConfigManager.getHandler(ReferenceCatalogConfig).parseArgs(args))
        options.update(ConfigManager.getHandler(ReferenceSamplePhotometryConfig).parseArgs(args))
        return options


ConfigManager.addHandler(ReferenceConfig)