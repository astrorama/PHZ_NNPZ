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
Created on: 10/11/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import os
import pathlib
from typing import Union

import numpy as np
from nnpz.exceptions import InvalidDimensionsException, InvalidAxisException, \
    UninitializedException, CorruptedFileException


class SedDataProvider(object):
    """
    Class for handling the SED data file of NNPZ format
    """

    def __init__(self, filename: Union[str, pathlib.Path]):
        """
        Creates a new instance for handling the given file.

        Raises:
            FileNotFoundException: If the file does not exist
        """
        self.__filename = filename
        try:
            self.__data = np.load(filename, mmap_mode='r') if os.path.exists(filename) else None
        except ValueError:
            raise CorruptedFileException()

        if self.__data is not None:
            if len(self.__data.shape) != 3:
                raise Exception('Expected an NdArray with three dimensions')
            if self.__data.shape[2] != 2:
                raise Exception('Expected an NdArray with the size of the last axis being 2')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not isinstance(self.__data, np.memmap):
            np.save(self.__filename, self.__data)

    def flush(self):
        """
        Write the changes to disk and re-load as a memory mapped file.
        To be used when this provider is full
        """
        if isinstance(self.__data, np.memmap):
            self.__data.flush()
        else:
            np.save(self.__filename, self.__data)
            self.__data = np.load(self.__filename, mmap_mode='r')

    def size(self) -> int:
        """
        Return the size on disk
        """
        return os.path.getsize(self.__filename) if os.path.exists(self.__filename) else 0

    def getKnots(self) -> int:
        """
        Return how many knots store this SED provider
        """
        return self.__data.shape[1] if self.__data is not None else 0

    def readSed(self, pos: int) -> np.ndarray:
        """
        Reads the data of an SED.

        Args:
            pos: The position of the SED in the file

        Returns:
            The data of the SED as a two dimensional numpy array of single
            precision floats. The first dimension has size same as the
            number of the knots and the second dimension has always size
            equal to two, with the first element representing the wavelength
            and the second the energy value.

        Raises:
            UninitializedException: If not initialized yet
        """
        if self.__data is None:
            raise UninitializedException()
        return self.__data[pos, :, :]

    def appendSed(self, data: Union[np.ndarray, list]) -> int:
        """
        Appends an SED to the end of the file.

        Args:
            data: The SED data as a two dimensional object, where the first
                dimension has size same as the number of the knots and the
                second dimension has always size equal to two, with the first
                element representing the wavelength and the second the energy
                value.

        Returns:
            The position in the file where the SED was added

        Raises:
            InvalidDimensionsException: If the dimensions of the given data
                object are incorrect.
            InvalidAxisException: If there are decreasing wavelength values
        """

        # First convert the data in a numpy array for easier handling
        data_arr = np.asarray(data, dtype='float32')

        # Check the dimensions
        if len(data_arr.shape) != 2:
            raise InvalidDimensionsException(
                'data must be a two dimensional array but it had {} dimensions'.format(
                    len(data_arr.shape)
                )
            )
        if data_arr.shape[1] != 2:
            raise InvalidDimensionsException(
                'data second dimension must be of size 2 but was {}'.format(data_arr.shape[1])
            )

        # Check that the wavelength axis does not have decreasing values
        if not np.all(data_arr[:-1, 0] <= data_arr[1:, 0]):
            raise InvalidAxisException('Wavelength axis must no have decreasing values')

        if self.__data is not None:
            self.__data = np.concatenate([self.__data, data_arr.reshape(1, -1, 2)], axis=0)
        else:
            self.__data = np.array(data_arr.reshape(1, -1, 2), copy=True)
        return len(self.__data) - 1
