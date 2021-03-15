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
Created on: 10/11/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import os
import pathlib
from typing import Union, Iterable

import numpy as np
from nnpz.exceptions import AlreadySetException, InvalidDimensionsException, \
    InvalidAxisException, UninitializedException, CorruptedFileException


class PdzDataProvider(object):
    """Class for handling the PDZ data file of NNPZ format"""

    def __init__(self, filename: Union[str, pathlib.Path]):
        """
        Creates a new instance for handling the given file.
        """
        self.__filename = filename
        try:
            self.__data = np.load(filename, mmap_mode='r') if os.path.exists(filename) else None
        except ValueError:
            raise CorruptedFileException()

        # Read from the file the redshift bins, if the header is there
        self.__redshift_bins = None
        if self.__data is not None:
            if len(self.__data.shape) != 2:
                raise CorruptedFileException('Expected an NdArray with two dimensions')
            if len(self.__data) > 0:
                self.__redshift_bins = self.__data[0, :]

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

    def setRedshiftBins(self, bins: np.ndarray):
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
        self.__data = self.__redshift_bins.reshape((1, -1))

    def getRedshiftBins(self) -> np.ndarray:
        """
        Returns the redshift bins of the PDZs in the file.

        Returns: A 1D numpy array of single precision floats with the redshift
            bins or None if the redhsift bins are not set yet
        """
        return self.__redshift_bins

    def readPdz(self, pos: int) -> np.ndarray:
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
        return self.__data[pos, :]

    def appendPdz(self, data: Union[np.ndarray, list]) -> Union[int, Iterable]:
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
        elif len(data_arr.shape) == 1:
            data_arr = data_arr.reshape(1, -1)

        if data_arr.shape[1] != len(self.__redshift_bins):
            raise InvalidDimensionsException(
                'PDZ data length differs from the redshift bins length')

        self.__data = np.concatenate([self.__data, data_arr], axis=0)
        if data_arr.shape[0] == 1:
            return len(self.__data) - 1
        return len(self.__data) - np.arange(data_arr.shape[0], 0, -1, dtype=np.int64)
