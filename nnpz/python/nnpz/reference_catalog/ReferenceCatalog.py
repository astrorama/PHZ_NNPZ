#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
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
Created on: 16/09/19
Author: Alejandro Alvarez Ayllon
"""
import numpy as np
from nnpz.exceptions import IdMismatchException, CorruptedFileException


class ReferenceCatalog(object):
    """
    Wraps a catalog with an API compatible with the native ReferenceSample.
    See Also: ReferenceSample.py
    """

    def __init__(self, ids, pdz, bins):
        """
        Constructor

        Args:
            ids:
                IDs of the objects in the reference catalog
            pdz:
                The probability distribution of each object in the catalog
            bins:
        """
        self.__ids = ids
        self.__pdz = pdz
        self.__bins = bins

    def size(self):
        """
        Returns the number of objects in the reference catalog
        """
        return len(self.__ids)

    def getIds(self):
        """
        Returns the IDs of the reference catalog objects
        """
        return self.__ids

    def getPdzData(self, obj_id):
        """
        Returns the PDZ data for the given reference catalog object.

        Args:
            obj_id: The ID of the object to retrieve the PDZ for

        Returns:
            None if the PDZ is not set for the given object, otherwise the data
            of the PDZ as a two dimensional numpy array of single precision
            floats. The first dimension has size same as the number of the knots
            and the second dimension has always size equal to two, with the
            first element representing the wavelength and the second the
            probability value.

        Raises:
            IdMismatchException: If there is no such ID in the reference sample
            CorruptedFileException: If the ID stored in the index file is
                different than the one stored in the PDZ data file
        """
        pdz = self.__pdz[self.__ids == obj_id]
        if len(pdz) == 0:
            raise IdMismatchException()
        if len(pdz) > 1:
            raise CorruptedFileException()
        return np.stack([self.__bins, pdz[0]]).T
