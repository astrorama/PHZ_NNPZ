"""
Created on: 10/07/18
Author: Florian Dubath
"""

from __future__ import division, print_function

from nnpz.config import ConfigManager
from nnpz.config.nnpz import TargetCatalogConfig
from nnpz.config.reference import ReferenceConfig
from nnpz.utils import Logging
from nnpz.photometry.SourceIndependantGalacticUnReddening import SourceIndependantGalacticUnReddening

logger = Logging.getLogger('Configuration')


class GalacticUnreddenerConfig(ConfigManager.ConfigHandler):

    def __init__(self):
        self.__galactic_absorption_unreddener = None
        self.__apply=False
        self.__parsed=False

    def __createGalacticUnreddener(self, args):
        if 'target_catalog_gal_ebv' in args:
            self.__apply=True;
        
            self._checkParameterExists('reference_sample_phot_filters', args)
    
            reference_config = ConfigManager.getHandler(ReferenceConfig).parseArgs(args)
            filter_order = args['reference_sample_phot_filters']
            filter_trans_map = reference_config['reference_filter_transmission']
            
            self.__galactic_absorption_unreddener = SourceIndependantGalacticUnReddening(
                    filter_trans_map, 
                    filter_order
            )
        self.__parsed=True


    def parseArgs(self, args):
        if not self.__parsed:
            self.__createGalacticUnreddener(args)
        return {'apply_galactic_absorption':self.__apply,'galactic_absorption_unreddener': self.__galactic_absorption_unreddener}


ConfigManager.addHandler(GalacticUnreddenerConfig)
