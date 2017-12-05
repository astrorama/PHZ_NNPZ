"""
Created on: 05/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from nnpz.photometry import SedPreprocessingInterface


class DefaultSedPreprocessing(SedPreprocessingInterface):
    """Performs no preprocessing of the SED templates."""


    def processSed(self, sed):
        return sed


    def type(self):
        return "ENERGY_FLUX"
