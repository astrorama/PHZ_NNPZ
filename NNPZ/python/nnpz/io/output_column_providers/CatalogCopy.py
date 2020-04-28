"""
Created on: 01/03/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from nnpz.io import OutputHandler


class CatalogCopy(OutputHandler.OutputColumnProviderInterface):


    def __init__(self, catalog, columns = None):
        self.__catalog = catalog
        if columns is not None:
            self.__columns = columns
        else:
            self.__columns = self.__catalog.colnames


    def addContribution(self, reference_sample_i, neighbor, flags):
        pass


    def getColumns(self):
        return [c for _,c in self.__catalog[self.__columns].columns.items()]
