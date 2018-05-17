"""
Created on: 30/01/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import abc
import os
import inspect
from astropy.table import Table

from nnpz.exceptions import *


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


    def __init__(self, filename):
        """Creates a new instance of CatalogReader.

        Args:
            filename: The file containing the catalog

        Raises:
            FileNotFoundException: If the given file does not exist
            WrongFormatException: If the file is not a catalog
        """

        if not os.path.exists(filename):
            raise FileNotFoundException('Missing file {}'.format(filename))

        try:
            self.__catalog = Table.read(filename)
        except:
            try:
                self.__catalog = Table.read(filename, format='ascii')
            except:
                raise WrongFormatException('Failed to read catalog {}'.format(filename))


    def get(self, property):
        """Returns the requested property of the catalog.

        Args:
            property: The property to return

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
        if inspect.isclass(property):
            property = property()

        if not isinstance(property, CatalogReader.CatalogPropertyInterface):
            raise WrongTypeException('property must implement the CatalogPropertyInterface')

        return property(self.__catalog)


    def getAsAstropyTable(self):
        """Returns the underlying astropy table"""

        return self.__catalog