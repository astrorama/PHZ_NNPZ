"""
Created on: 28/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from nnpz.utils import Logging
from nnpz.config import ConfigManager

logger = Logging.getLogger('Configuration')


class ReferenceUniqueConfig(ConfigManager.ConfigHandler):

    def parseArgs(self, args):
        if 'reference_sample_dir' in args and 'reference_catalog' in args:
            logger.error('Only one of reference_sample_dir and reference_catalog options can be set')
            exit(-1)
        if 'reference_sample_dir' not in args and 'reference_catalog' not in args:
            logger.error('One of reference_sample_dir and reference_catalog options must be set')
            exit(-1)
        return {}


ConfigManager.addHandler(ReferenceUniqueConfig)