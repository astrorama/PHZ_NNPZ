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
import os
import pathlib
from typing import Union

import numpy as np
from nnpz.exceptions import CorruptedFileException, UninitializedException, \
    InvalidDimensionsException


class MontecarloDataProvider(object):
    """
    Can be used to model a PDF of the reference objects as a set of random samples.
    Used for Intermediate Bands and Physical parameters.
    """

    def __init__(self, filename: Union[str, pathlib.Path]):
        """
        Creates a new instance for handling the given file.
        """
        self.__filename = filename
        try:
            self.__data = np.load(filename, mmap_mode='r') if os.path.exists(filename) else None
        except ValueError:
            raise CorruptedFileException()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.__data is not None and not isinstance(self.__data, np.memmap):
            np.save(self.__filename, self.__data)
            self.__data = np.load(self.__filename, mmap_mode='r')

    def flush(self):
        """
        Write the changes to disk and re-load as a memory mapped file.
        To be used when this provider is full
        """
        if self.__data is None:
            return
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

    def read(self, pos: int) -> np.ndarray:
        """
        Read the entry at the given position

        Raises:
            UninitializedException: If the data has not been set
        """
        if self.__data is None:
            raise UninitializedException()
        return self.__data[pos]

    def append(self, data: np.ndarray):
        """
        Append new data. On the first call, anything can be passed. On the following,
        there will be a check on the dimensionality, so all entries match.

        Args:
            data: np.ndarray
                New entries with the shape n.entries, n.samples, n.coordinates
                If there are only two axes, it will be assumed to be a single entry and
                reshaped accordingly
        Raises:
            InvalidDimensionsException: If the dimensionality of the data does not match
        """
        if len(data.shape) not in (1, 2):
            raise InvalidDimensionsException(
                f'Expected one or two axes, got {len(data.shape)}'
            )
        elif len(data.shape) == 1:
            data = data.reshape(1, *data.shape)

        if self.__data is None:
            self.__data = np.array(data, copy=True)
            return np.arange(0, data.shape[0])
        if self.__data.shape[1:] != data.shape[1:]:
            raise InvalidDimensionsException(
                f'Dimensionality mismatch: {self.__data.shape[1:]} vs {data.shape[1:]}'
            )
        self.__data = np.concatenate([self.__data, data], axis=0)
        return len(self.__data) - np.arange(data.shape[0], 0, -1, dtype=np.int64)
