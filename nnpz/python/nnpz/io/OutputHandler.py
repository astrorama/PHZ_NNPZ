#
# Copyright (C) 2012-2021 Euclid Science Ground Segment
#
# This library is free software; you can redistribute it and/or modify it under the terms of
# the GNU Lesser General Public License as published by the Free Software Foundation;
# either version 3.0 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this library;
# if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301 USA
#

"""
Created on: 01/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import abc
import os

from astropy.io import fits
from astropy.table import Table

from nnpz.utils.fits import tableToHdu, columnsToFitsColumn


class OutputHandler(object):
    """
    Handles the generation of the output properties from the found reference objects
    and the given configuration
    """

    class OutputColumnProviderInterface(object):
        """
        This interface must be implemented by output properties that generate a column
        """
        __metaclass__ = abc.ABCMeta

        @abc.abstractmethod
        def addContribution(self, reference_sample_i, neighbor, flags):
            """
            This method is called for each positive pair reference/neighbor
            Args:
                reference_sample_i: int
                    *Index* of the reference sample object
                neighbor: nnpz.framework.NeighborSet.Neighbor
                    Neighbor properties
                flags: NnpzFlag
                    A flag object for this pair, if the provider needs to set any
            """

        @abc.abstractmethod
        def getColumns(self):
            """
            Returns: list of astropy.table.Column
            """

    class OutputExtensionTableProviderInterface(object):
        """
        This interface must be implemented by output properties that generate an additional
        table
        """
        __metaclass__ = abc.ABCMeta

        @abc.abstractmethod
        def addContribution(self, reference_sample_i, neighbor, flags):
            """
            This method is called for each positive pair reference/neighbor
            Args:
                reference_sample_i: int
                    *Index* of the reference sample object
                neighbor: nnpz.framework.NeighborSet.Neighbor
                    Neighbor properties
                flags: NnpzFlag
                    A flag object for this pair, if the provider needs to set any
            """

        @abc.abstractmethod
        def getExtensionTables(self):
            """
            Returns:
                A dictionary with additional astropy.Table, where the key is the table name
            """
            pass

    class HeaderProviderInterface(object):
        """
        This interface must be implemented by output properties that generate a table header
        """
        __metaclass__ = abc.ABCMeta

        @abc.abstractmethod
        def getHeaderKeywords(self):
            """
            Returns:
                 A map with keys the keyword names and values the header values.
            """

    def __init__(self):
        self.__column_providers = []
        self.__hdu_providers = []
        self.__header_providers = []

    def addColumnProvider(self, provider):
        """
        Register a new column provider
        Args:
            provider:  OutputColumnProviderInterface
        """
        self.__column_providers.append(provider)

    def addExtensionTableProvider(self, provider):
        """
        Register a new table provider
        Args:
            provider: OutputExtensionTableProviderInterface
        """
        self.__hdu_providers.append(provider)

    def addHeaderProvider(self, provider):
        """
        Register a new header provider
        Args:
            provider: HeaderProviderInterface
        """
        self.__header_providers.append(provider)

    def addContribution(self, reference_sample_i, neighbor, flags):
        """
        This method is to be called for each positive pair reference/neighbor.
        It will be forwarded to the registered output providers.
        Args:
            reference_sample_i: int
                *Index* of the reference sample object
            neighbor: nnpz.framework.NeighborSet.Neighbor
                Neighbor properties
            flags: NnpzFlag
                A flag object for this pair, if the provider needs to set any
        """
        for col_provider in self.__column_providers:
            col_provider.addContribution(reference_sample_i, neighbor, flags)
        for hdu_provider in self.__hdu_providers:
            hdu_provider.addContribution(reference_sample_i, neighbor, flags)

    def save(self, filename):
        """
        Write the output catalog
        Args:
            filename: str or path
                Output file path
        """
        hdu_list = []

        # Primary hdu
        hdr = fits.Header()
        hdr['COMMENT'] = 'Generated by nnpz'
        hdu_list.append(fits.PrimaryHDU(header=hdr))

        # Table with the results
        columns = []
        for prov in self.__column_providers:
            columns.extend(prov.getColumns())
        hdr = fits.Header()
        hdr['COMMENT'] = 'Generated by nnpz'
        for prov in self.__header_providers:
            hdr.update(prov.getHeaderKeywords())
        hdu_list.append(fits.BinTableHDU(Table(columns), header=hdr))

        # Extensions
        for hdu_provider in self.__hdu_providers:
            for name, table in hdu_provider.getExtensionTables().items():
                ext_hdu = fits.BinTableHDU(table)
                ext_hdu.name = name
                hdu_list.append(ext_hdu)

        hdul = fits.HDUList(hdu_list)

        if os.path.exists(filename):
            os.remove(filename)
        hdul.writeto(filename)
