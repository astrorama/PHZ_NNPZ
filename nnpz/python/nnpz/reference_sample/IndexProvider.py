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
Created on: 16/11/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import os
import pathlib
from collections import namedtuple
from typing import Union

import numpy as np
from ElementsKernel import Logging
from nnpz.exceptions import CorruptedFileException, DuplicateIdException
from numpy.lib import recfunctions as rfn

logger = Logging.getLogger(__name__)


class IndexProvider(object):
    """
    Class for handling the index table of the NNPZ format
    """

    ObjectLocation = namedtuple('ObjectLocation', ['file', 'offset'])

    def __checkLayout(self):
        """
        Make sure the layout is contiguous on disk
        """
        pairs = {}
        for field in self.__data.dtype.fields:
            if field.endswith('_file') or field.endswith('_offset'):
                dataset = field.split('_')[0]
                if not dataset in pairs:
                    pairs[dataset] = []
                pairs[dataset].append(field)

        for key, pair in pairs.items():
            file_field, offset_field = pair
            order = np.argsort(self.__data, order=(file_field, offset_field))
            if (np.diff(order) < 0).any():
                logger.warning(
                    'The index for the provider "{}" does not follow the physicial layout'.format(key)
                )

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
                self.__data = np.load(filename, mmap_mode='r+')
            except ValueError:
                raise CorruptedFileException()
        else:
            self.__data = np.array([], dtype=[('id', np.int64)])

        if not self.__data.dtype.names:
            raise CorruptedFileException('Expected a structured array')
        if 'id' not in self.__data.dtype.names:
            raise CorruptedFileException('Missing ids')
        if not all(map(lambda d: d != np.int64, self.__data.dtype.fields.values())):
            raise CorruptedFileException('Expected 64 bits integers')

        # Make sure the disk layout is contiguous
        self.__checkLayout()

        # Create a map for easier search and at the same time check if we have
        # duplicates. The map values are (i, sed_file, sed_pos, pdz_file, pdz_pos),
        # where i is the index of the ID.
        self.__index_map = {}
        for i in range(len(self.__data)):
            row = self.__data[i]
            if row['id'] in self.__index_map:
                raise DuplicateIdException('Duplicate ID {}'.format(row['id']))
            self.__index_map[row['id']] = i

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
            self.__data = np.load(self.__filename, mmap_mode='r+')

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
        return self.__data['id']

    def getFiles(self, key: str) -> set:
        """
        Returns a list of short unsigned integers with the file indices for the given key
        """
        try:
            files = set(self.__data[f'{key}_file'])
            if -1 in files:
                files.remove(-1)
            return files
        except ValueError:
            return set()

    def get(self, obj_id: int, key: str) -> 'ObjectLocation':
        """
        Returns the position for a given ID.

        Args:
            obj_id: The ID of the object to retrieve the info for
            key: The index key

        Returns:
            A named tuple with the following:
                - The file index
                - The file offset
        """
        if obj_id not in self.__index_map:
            return None
        i = self.__index_map[obj_id]
        try:
            loc = IndexProvider.ObjectLocation(*self.__data[i][[f'{key}_file', f'{key}_offset']])
            if loc.file == -1:
                loc = None
            return loc
        except KeyError:
            return None

    def _addKey(self, key: str):
        """
        Add a new key to the index
        """
        self.__data = np.array(rfn.append_fields(
            self.__data,
            data=np.full((2, self.__data.shape[0]), -1),
            names=[f'{key}_file', f'{key}_offset'],
            dtypes=np.int64
        ), copy=True)

    def clear(self, key: str):
        """
        If key is part of the index, clear the values
        """
        if f'{key}_file' in self.__data.dtype.names:
            self.__data[f'{key}_file'] = -1
            self.__data[f'{key}_offset'] = -1

    def add(self, obj_id: int, key: str, location: ObjectLocation):
        """
        Add a new entry to the index

        Args:
            obj_id: The ID of the object to retrieve the info for
            key: The index key
            location: The location of the data

        Raises:

        """
        if f'{key}_file' not in self.__data.dtype.names:
            self._addKey(key)

        try:
            i = self.__index_map[obj_id]
        except KeyError:
            entry = np.full(shape=1, fill_value=-1, dtype=self.__data.dtype)
            entry['id'] = obj_id
            self.__data = np.concatenate([self.__data, entry], axis=0)
            i = self.__index_map[obj_id] = self.__data.shape[0] - 1

        entry = self.__data[i]
        if entry[f'{key}_file'] != -1:
            raise DuplicateIdException(
                'Index already contains {} (file {})'.format(obj_id, entry[f'{key}_file']))

        entry[f'{key}_file'] = location.file
        entry[f'{key}_offset'] = location.offset

    def bulkAdd(self, other: np.ndarray):
        """
        Concatenate a whole other index
        """
        if not other.dtype.names:
            raise ValueError('Expected a structured array')
        if 'id' not in other.dtype.names:
            raise ValueError('Missing id field')
        if not all(map(lambda d: d != np.int64, other.dtype.fields.values())):
            raise CorruptedFileException('Expected 64 bits integers')

        this = self.__data

        # Get set of unique IDs
        unique_ids = np.unique(np.concatenate([this['id'], other['id']]))
        unique_ids.sort()

        # Get set of columns
        columns = set(this.dtype.names)
        columns.update(other.dtype.names)
        columns.discard('id')
        columns = list(sorted(columns))

        # Pre-allocate destination
        destination = np.full(
            len(unique_ids), -1,
            dtype=[('id', np.int64)] + list(map(lambda c: (c, np.int64), columns))
        )

        # Copy IDs over
        destination['id'] = unique_ids

        # Copy columns from both sides, checking for duplicates
        this_idx = np.searchsorted(destination['id'], this['id'])
        other_idx = np.searchsorted(destination['id'], other['id'])
        for c in columns:
            # From self
            if c in this.dtype.names:
                destination[c][this_idx] = this[c]
            # From the other
            if c in other.dtype.names:
                already_set = destination[c][other_idx] != -1
                if np.any(already_set):
                    raise DuplicateIdException(destination['id'][other_idx[already_set]])
                destination[c][other_idx] = other[c]

        self.__data = destination
        for i in range(self.__data.shape[0]):
            self.__index_map[self.__data['id'][i]] = i

    @property
    def raw(self):
        return self.__data

    def getIdsForFile(self, file_id: int, key: str):
        """
        Returns:
            The list of object IDs stored on a given file, following the physical order
        """
        idx_mask = self.__data[f'{key}_file'] == file_id
        idx_masked = self.__data[idx_mask]
        disk_order = np.argsort(idx_masked[f'{key}_offset'])
        return idx_masked[disk_order]['id']
