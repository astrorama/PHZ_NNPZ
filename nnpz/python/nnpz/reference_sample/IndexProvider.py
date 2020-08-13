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
Created on: 16/11/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import os
import pathlib
from collections import namedtuple
from typing import Union

import numpy as np
from nnpz.exceptions import CorruptedFileException, DuplicateIdException


class IndexProvider(object):
    """
    Class for handling the index table of the NNPZ format
    """

    ObjectLocation = namedtuple('ObjectLocation', ['file', 'offset'])

    def __init__(self, filename: Union[str, pathlib.Path]):
        """
        Creates a new instance for handling the given file.

        Args:
            filename: string, or pathlib.Path

        Raises:
            FileNotFoundException: If the file does not exist
            CorruptedFileException: If the file is malformed
            IdMismatchException: If there are duplicate IDs
            InvalidPositionException: If a SED or PDZ position is less than -1
        """
        self.__filename = filename

        if os.path.exists(filename):
            try:
                self.__data = np.load(filename, mmap_mode='r')
            except ValueError:
                raise CorruptedFileException()
        else:
            self.__data = np.zeros(shape=(0, 3), dtype=np.int64)

        if self.__data.dtype != np.int64:
            raise CorruptedFileException('Expected 64 bits integers')
        if len(self.__data.shape) != 2:
            raise CorruptedFileException('Expected an array with two dimensions')
        if self.__data.shape[1] != 3:
            raise CorruptedFileException('The second dimension is expected to be of size 3')

        # Create a map for easier search and at the same time check if we have
        # duplicates. The map values are (i, sed_file, sed_pos, pdz_file, pdz_pos),
        # where i is the index of the ID.
        self.__index_map = {}
        self.__files = set()
        for row in self.__data:
            if row[0] in self.__index_map:
                raise DuplicateIdException('Duplicate ID {}'.format(row[0]))
            self.__index_map[row[0]] = IndexProvider.ObjectLocation(row[1], row[2])
            self.__files.add(row[1])

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
        if not isinstance(self.__data, np.memmap):
            np.save(self.__filename, self.__data)
            self.__data = np.load(self.__filename, mmap_mode='r')

    def __len__(self) -> int:
        """
        Returns the number of objects in the index
        """
        return len(self.__index_map)

    def size(self) -> int:
        """
        Returns the disk size of the index
        """
        return os.path.getsize(self.__filename) if os.path.exists(self.__filename) else 0

    def getIds(self) -> list:
        """
        Returns a list of long integers with the IDs
        """
        return self.__data[:, 0]

    def getFiles(self) -> set:
        """
        Returns a list of short unsigned integers with the SED file indices
        """
        return self.__files

    def get(self, obj_id: int) -> ObjectLocation:
        """
        Returns the position for a given ID.

        Args:
            obj_id: The ID of the object to retrieve the info for

        Returns:
            A named tuple with the following:
                - The file index
                - The file offset
        """
        if obj_id not in self.__index_map:
            return None
        return self.__index_map[obj_id]

    def add(self, obj_id: int, location: ObjectLocation):
        """
        Add a new entry to the index

        Args:
            obj_id: The ID of the object to retrieve the info for
            location: The location of the data

        Raises:

        """
        if obj_id in self.__index_map:
            raise DuplicateIdException('Index already contains {}'.format(obj_id))
        self.__index_map[obj_id] = location
        self.__files.add(location.file)
        entry = np.array([[obj_id, location.file, location.offset]], dtype=np.int64)
        self.__data = np.concatenate([self.__data, entry], axis=0)

    def bulkAdd(self, other: np.ndarray):
        """
        Concatenate a whole other index
        """
        if np.any(np.in1d(other[:, 0], self.__data[:, 0])):
            raise DuplicateIdException()
        self.__data = np.concatenate([self.__data, other], axis=0)
        for row in other:
            self.__index_map[row[0]] = IndexProvider.ObjectLocation(row[1], row[2])
