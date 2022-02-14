#
# Copyright (C) 2012-2022 Euclid Science Ground Segment
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


import os
import pathlib
from typing import Union

import numpy as np
from nnpz.exceptions import CorruptedFileException, InvalidAxisException, \
    InvalidDimensionsException, UninitializedException


class SedDataProvider:
    """
    Class for handling the SED data file of NNPZ format
    """

    def __try_load(self):
        """
        Initialize internal members and perform some checks if the datafile exists
        """
        if not os.path.exists(self.__filename):
            return
        try:
            data = np.load(self.__filename, mmap_mode='r')
            if len(data.shape) != 3:
                raise CorruptedFileException('Expected an NdArray with three dimensions')
            if data.shape[2] != 2:
                raise CorruptedFileException(
                    'Expected an NdArray with the size of the last axis being 2')
            self.__knots = data.shape[1]
            self.__entries = data.shape[0]
        except ValueError:
            raise CorruptedFileException()

    def __init__(self, filename: Union[str, pathlib.Path], cache_size: int = 5000):
        """
        Creates a new instance for handling the given file.
        Args:
            filename:
                Data file
            cache_size:
                Number of entries to keep in memory in any given time
        """
        self.__filename = filename
        self.__cache_size = cache_size
        self.__cache_line = None
        self.__cache = None
        self.__entries = 0
        self.__full = None
        self.__knots = 0
        self.__try_load()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()

    def flush(self):
        """
        Write the changes to disk and re-load as a memory mapped file.
        To be used when this provider is full
        """
        if self.__full is not None:
            np.save(self.__filename, self.__full)
            self.__full = None

    def size(self) -> int:
        """
        Return the size on disk
        """
        return os.path.getsize(self.__filename) if os.path.exists(self.__filename) else 0

    def get_knots_size(self) -> int:
        """
        Return how many knots store this SED provider
        """
        return self.__knots

    def __from_cache(self, pos: int) -> int:
        """
        Return the offset within the cache
        """
        if pos > self.__entries:
            raise InvalidAxisException('Position {} for {} entries'.format(pos, self.__entries))
        cache_line = pos // self.__cache_size
        if cache_line != self.__cache_line:
            self.__cache_line = cache_line
            self.__cache = np.load(self.__filename, mmap_mode='r')
        return pos

    def read_sed(self, pos: int) -> np.ndarray:
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
        if self.__knots == 0:
            raise UninitializedException()
        if self.__full is not None:
            return self.__full[pos]
        offset = self.__from_cache(pos)
        return self.__cache[offset]

    def append_sed(self, data: Union[np.ndarray, list]) -> int:
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
        if len(data_arr.shape) not in (2, 3):
            raise InvalidDimensionsException(
                'SED data must be a two or three dimensional array but it had {} dimensions'.format(
                    len(data_arr.shape)
                )
            )

        if data_arr.shape[-1] != 2:
            raise InvalidDimensionsException(
                'SED data last dimension must be of size 2 but was {}'.format(data_arr.shape[1])
            )

        if len(data_arr.shape) == 2:
            data_arr = data_arr.reshape(1, -1, 2)

        # Check that the wavelength axis does not have decreasing values
        if not np.all(data_arr[:, :-1, 0] <= data_arr[:, 1:, 0]):
            raise InvalidAxisException('Wavelength axis must no have decreasing values')

        if self.__full is None and os.path.exists(self.__filename):
            self.__full = np.load(self.__filename)
        if self.__full is None:
            self.__full = np.array(data_arr, copy=True)
            self.__knots = data_arr.shape[1]
        else:
            self.__full = np.concatenate([self.__full, data_arr], axis=0)

        self.__entries = len(self.__full)
        if data_arr.shape[0] == 1:
            return self.__entries - 1
        return self.__entries - np.arange(data_arr.shape[0], 0, -1, dtype=np.int64)
