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
Created on: 30/01/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import abc
import os
import inspect

from astropy.io.registry import IORegistryError
from astropy.table import Table
from nnpz.exceptions import FileNotFoundException, WrongFormatException, WrongTypeException


class CatalogReader(object):
    """Utility class for reading catalog files.

    This class does not know about the semantics of the catalog columns. It
    delegates this task to implementations of the CatalogPropertyInterface.
    The default NNPZ implementations of this interface can be found in the
    nnpz.io.catalog_properties module.
    """

    class CatalogPropertyInterface(object):
        """Interface representing a catalog property"""
        __metaclass__ = abc.ABCMeta

        @abc.abstractmethod
        def __call__(self, catalog):
            """Must be implemented to return the property of the catalog.

            Args:
                catalog: An astropy table representing the catalog

            Returns:
                The property for the catalog

            Note that the property is for the full catalog, so when there is a
            value per object it is the responsibility of the implementation to
            return a list.
            """
            return

    def __init__(self, filename, hdu=1):
        """Creates a new instance of CatalogReader.

        Args:
            filename: The file containing the catalog
            hdu: For FITS file, the HDU to read

        Raises:
            FileNotFoundException: If the given file does not exist
            WrongFormatException: If the file is not a catalog
        """

        if not os.path.exists(filename):
            raise FileNotFoundException('Missing file {}'.format(filename))

        try:
            self.__catalog = Table.read(filename, hdu=hdu)
        except (IORegistryError, ValueError):
            try:
                self.__catalog = Table.read(filename, format='ascii')
            except ValueError:
                raise WrongFormatException('Failed to read catalog {}'.format(filename))

    def get(self, prop):
        """Returns the requested property of the catalog.

        Args:
            prop: The property to return

        Returns:
            The requested property

        Raises:
            WrongTypeException: If the given property is not implementing the
                CatalogPropertyInterface interface
            The exceptions that might be raised by the property

        Note that this method can receive both an instance of the property or
        the property class. In the second case, an instance is constructed using
        a constructor without arguments.

        """

        # If we got the class create the instance using the default constructor
        if inspect.isclass(prop):
            prop = prop()

        if not isinstance(prop, CatalogReader.CatalogPropertyInterface):
            raise WrongTypeException('property must implement the CatalogPropertyInterface')

        return prop(self.__catalog)

    def size(self):
        return len(self.__catalog)

    def getAsAstropyTable(self):
        """Returns the underlying astropy table"""

        return self.__catalog
