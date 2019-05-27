"""
Created on: 28/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from nnpz.utils import Logging
from nnpz.config import ConfigManager
from nnpz import ReferenceSample

logger = Logging.getLogger('Configuration')


class ReferenceSampleConfig(ConfigManager.ConfigHandler):

    def __init__(self):
        self.__sample = None

    def __createSample(self, args):
        if 'reference_sample_dir' in args:
            sample_dir = args['reference_sample_dir']
            logger.info('Reading reference sample from {}...'.format(sample_dir))
            self.__sample = ReferenceSample(sample_dir)
            logger.info('Reading reference sample done')

    def parseArgs(self, args):
        if self.__sample is None:
            self.__createSample(args)

        if self.__sample is not None:
            return {'reference_sample' : self.__sample,
                    'reference_ids' : self.__sample.getIds()}
        else:
            return {}


ConfigManager.addHandler(ReferenceSampleConfig)