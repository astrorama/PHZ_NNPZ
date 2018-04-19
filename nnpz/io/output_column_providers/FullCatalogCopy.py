"""
Created on: 01/03/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from nnpz.io import OutputHandler


class FullCatalogCopy(OutputHandler.OutputColumnProviderInterface):


    def __init__(self, catalog):
        self.__catalog = catalog


    def addContribution(self, reference_sample_i, catalog_i, weight):
        pass


    def getColumns(self):
        return [c for _,c in self.__catalog.columns.items()]
