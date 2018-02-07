"""
Created on: 01/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import abc
import os

from astropy.table import Table


class OutputHandler(object):


    class OutputColumnProviderInterface(object):
        __metaclass__ = abc.ABCMeta

        @abc.abstractmethod
        def addContribution(self, reference_sample_i, catalog_i, weight):
            pass

        @abc.abstractmethod
        def getColumns(self):
            pass


    def __init__(self):
        self.__providers = []


    def addColumnProvider(self, provider):
        self.__providers.append(provider)


    def addContribution(self, reference_sample_i, catalog_i, weight):
        for p in self.__providers:
            p.addContribution(reference_sample_i, catalog_i, weight)


    def save(self, filename):
        out = Table()
        for prov in self.__providers:
            for col in prov.getColumns():
                out.add_column(col)

        if os.path.exists(filename):
            os.remove(filename)
        out.write(filename, format='fits')
