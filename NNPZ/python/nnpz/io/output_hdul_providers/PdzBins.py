"""
Created on: 19/04/2018
Author: Alejandro Alvarez Ayllon
"""
import numpy as np
from astropy.table import Table

from nnpz.io.OutputHandler import OutputHandler


class PdzBins(OutputHandler.OutputExtensionTableProviderInterface):
    """
    Generates an HDUL with the PDZ bins
    """

    def __init__(self, pdz_provider):
        self.__pdz_provider = pdz_provider

    def addContribution(self, reference_sample_i, catalog_i, weight, flags):
        pass

    def getExtensionTables(self):
        bins = self.__pdz_provider.getPdzBins()
        return {
            'PDZBins': Table({
                'Redshift': bins,
            })
        }
