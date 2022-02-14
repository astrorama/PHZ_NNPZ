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
from typing import Iterable, Union

import numpy as np
from nnpz.exceptions import AlreadySetException, CorruptedFileException, InvalidAxisException, \
    InvalidDimensionsException, UninitializedException


class PdzDataProvider:
    """
    Class for handling the PDZ data file of NNPZ format
    """

    def __try_load(self):
        """
        Initialize internal members and perform some checks if the datafile exists
        """
        if not os.path.exists(self.__filename):
            return
        try:
            data = np.load(self.__filename, mmap_mode='r')
            if len(data.shape) != 2:
                raise CorruptedFileException('Expected an NdArray with two dimensions')
            if len(data) > 0:
                self.__redshift_bins = data[0]
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
        self.__redshift_bins = None
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

    def set_redshift_bins(self, bins: np.ndarray):
        """
        Sets the redshift bins of the PDZs in the file.

        Args:
            bins: The redshift bins values as a 1D numpy array of single
                precision floats

        Raises:
            InvalidDimensionsException: If the given bins array is not 1D
            InvalidAxisException: If the given bins values are not strictly
                increasing
            AlreadySetException: If the redhsift bins of the file are already set
        """

        if self.__redshift_bins is not None:
            raise AlreadySetException('PDZ redshift bins are already set')

        if len(bins.shape) != 1:
            raise InvalidDimensionsException('The bins array must be 1D')

        if np.any(bins[:-1] >= bins[1:]):
            raise InvalidAxisException('Redshift bins must be strictly increasing')

        self.__redshift_bins = np.asarray(bins, dtype=np.float32)
        self.__full = self.__redshift_bins.reshape((1, -1))

    def get_redshift_bins(self) -> np.ndarray:
        """
        Returns the redshift bins of the PDZs in the file.

        Returns: A 1D numpy array of single precision floats with the redshift
            bins or None if the redhsift bins are not set yet
        """
        return self.__redshift_bins

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

    def read_pdz(self, pos: int) -> np.ndarray:
        """
        Reads a PDZ from the given position.

        Args:
            pos: The position of the PDZ in the file

        Returns: A tuple with the following:
            - The data of the PDZ as a one dimensional numpy array of single
                precision floats.

        Raises:
            UninitializedException: If the redshift bins are not set
        """
        if self.__redshift_bins is None:
            raise UninitializedException()
        if self.__full is not None:
            return self.__full[pos]
        offset = self.__from_cache(pos)
        return self.__cache[offset]

    def append_pdz(self, data: Union[np.ndarray, list]) -> Union[int, Iterable]:
        """
        Appends a PDZ to the end of the file.

        Args:
            data: The PDZ data as a single dimensional object

        Returns:
            The position in the file where the PDZ was added

        Raises:
            InvalidDimensionsException: If the dimensions of the given data object are incorrect
            UninitializedException: If the redshift bins are not set
        """

        if self.__redshift_bins is None:
            raise UninitializedException()

        # Convert the data in a numpy array
        data_arr = np.asarray(data, dtype=np.float32)
        if len(data_arr.shape) > 2:
            raise InvalidDimensionsException('PDZ data must be a 1D or 2D array')
        if len(data_arr.shape) == 1:
            data_arr = data_arr.reshape(1, -1)

        if data_arr.shape[1] != len(self.__redshift_bins):
            raise InvalidDimensionsException(
                'PDZ data length differs from the redshift bins length')

        if self.__full is None and os.path.exists(self.__filename):
            self.__full = np.load(self.__filename)
        if self.__full is None:
            self.__full = np.array(data_arr, copy=True)
        else:
            self.__full = np.concatenate([self.__full, data_arr], axis=0)

        self.__entries = len(self.__full)
        if data_arr.shape[0] == 1:
            return self.__entries - 1
        return self.__entries - np.arange(data_arr.shape[0], 0, -1, dtype=np.int64)
