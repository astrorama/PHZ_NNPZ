"""
Created on: 10/07/18
Author: Florian Dubath
"""

from __future__ import division, print_function

from nnpz.config import ConfigManager, ReferenceConfig, TargetCatalogConfig
from nnpz.utils import Logging
from nnpz.photometry.SourceIndependantGalacticUnReddening import SourceIndependantGalacticUnReddening

logger = Logging.getLogger('Configuration')


class GalacticUnreddenerConfig(ConfigManager.ConfigHandler):

    def __init__(self):
        pass

    def __createGalacticUnreddener(self, args):
        self._checkParameterExists('reference_sample_phot_filters', args)

        reference_config = ConfigManager.getHandler(ReferenceConfig).parseArgs(args)
        filter_order = args['reference_sample_phot_filters']
        filter_trans_map = reference_config['reference_filter_transmission']
        
        self.__galactic_absorption_unreddener = SourceIndependantGalacticUnReddening(
                filter_trans_map, 
                filter_order
        )



    def parseArgs(self, args):
        if self.__createGalacticUnreddener is None:
            self.__createGalacticUnreddener(args)
        return {'galactic_absorption_unreddener': self.__galactic_absorption_unreddener}


ConfigManager.addHandler(GalacticUnreddenerConfig)
