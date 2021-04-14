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
from nnpz.utils.fits import npDtype2FitsTForm, shape2FitsTDim


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
        def getColumnDefinition(self):
            """
            Returns:
                List of triplets (name, dtype, shape) that define the output that the
                provider generates. Alternative, a tuple can be used, skipping the shape.
                The caller will assume the output is scalar.
            Notes:
                Do *not* account for the catalog size on the shape.
            """
            raise NotImplementedError(self)

        @abc.abstractmethod
        def setWriteableArea(self, output_area):
            """
            This method will be called once the output area is allocated. It is called
            *before* addContribution. The provider can, but is not forced to, use the output
            buffer as a working area (i.e. for accumulating before normalizing)
            Args:
                output_area:
                    A structured array (accesible by name)
            """
            raise NotImplementedError(self)

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
            raise NotImplementedError(self)

        def fillColumns(self):
            """
            If the OutputHandler needs to do some final massaging of the data (i.e.
            normalization), or can only generate the output when all contributions have
            been added, this is the moment to do so.
            """
            raise NotImplementedError(self)

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
        self.__output = None

    def initialize(self, nrows: int):
        """
        Setup output area
        Args:
            nrows: int
                Number of target objects
        """
        # Build specs
        col_spec = []
        for col_provider in self.__column_providers:
            col_defs = col_provider.getColumnDefinition()
            for col_def in col_defs:
                name, dtype, shape = col_def if len(col_def) == 3 else col_def + (1,)
                # npDtype2FitsTForm expects the shape to account for the row "dimension"
                shape = (1,) + shape if isinstance(shape, tuple) else (1, shape,)
                col_spec.append(fits.Column(name=name, format=npDtype2FitsTForm(dtype, shape),
                                            dim=shape2FitsTDim(shape)))
        # Allocate
        self.__output = fits.BinTableHDU.from_columns(
            col_spec, nrows=nrows,
            header=fits.Header({'COMMENT': 'Generated by nnpz'})
        )
        # Now go back again and tell the handlers where to write
        for col_provider in self.__column_providers:
            col_provider.setWriteableArea(self.__output.data)

    def addColumnProvider(self, provider):
        """
        Register a new column provider
        Args:
            provider:  OutputColumnProviderInterface
        """
        assert self.__output is None
        self.__column_providers.append(provider)

    def addExtensionTableProvider(self, provider):
        """
        Register a new table provider
        Args:
            provider: OutputExtensionTableProviderInterface
        """
        assert self.__output is None
        self.__hdu_providers.append(provider)

    def addHeaderProvider(self, provider):
        """
        Register a new header provider
        Args:
            provider: HeaderProviderInterface
        """
        assert self.__output is None
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
        assert self.__output is not None
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
        if os.path.exists(filename):
            os.remove(filename)

        # Finish with the output data
        assert self.__output is not None
        for col_prov in self.__column_providers:
            col_prov.fillColumns()

        for prov in self.__header_providers:
            self.__output.header.update(prov.getHeaderKeywords())

        # Open file
        hdul = fits.open(filename, mode='append')

        # Primary hdu
        hdul.append(fits.PrimaryHDU(header=fits.Header({'COMMENT': 'Generated by nnpz'})))

        # Main results
        hdul.append(self.__output)

        # Extensions
        for hdu_provider in self.__hdu_providers:
            for name, table in hdu_provider.getExtensionTables().items():
                ext_hdu = fits.BinTableHDU(table)
                ext_hdu.name = name
                hdul.append(ext_hdu)

        hdul.close()
