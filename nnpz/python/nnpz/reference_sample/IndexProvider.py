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
import numpy as np

from nnpz.exceptions import *


class IndexProvider(object):
    """Class for handling the index table of the NNPZ format"""



    def __checkPositionValidity(self, pos_list, name):

        # Create a numpy array for easier handling
        pos_list = np.asarray(pos_list, dtype=np.int64)

        # Check if any value is less than -1
        if np.any(pos_list < -1):
            raise InvalidPositionException('Negative ' + name + ' position values')

        # Find the -1 entries
        null_indices = np.where(pos_list == -1)[0]

        # If we have a -1 entry check that all the next values are also -1 and trim them away
        if len(null_indices) > 0:
            if len(null_indices) < len(pos_list) - null_indices[0]:
                raise InvalidPositionException('Positive ' + name + 'position values after first -1')
            pos_list = pos_list[:null_indices[0]]

        # Now that we have removed the -1 values, check that it is strictly increasing
        if np.any(pos_list[:-1] >= pos_list[1:]):
            raise InvalidPositionException(name + ' positions are not strictly increasing')


    def __init__(self, filename):
        """Creates a new instance for handling the given file.

        Raises:
            FileNotFoundException: If the file does not exist
            CorruptedFileException: If the size of the file is not divisible by
                28, which is the length of the info for each object
            IdMismatchException: If there are duplicate IDs
            InvalidPositionException: If a SED or PDZ position is less than -1
        """
        self.__filename = filename

        if not os.path.exists(filename):
            raise FileNotFoundException(filename)

        # Check that the file size is consistent
        if not os.path.getsize(filename) % 28 == 0:
            raise CorruptedFileException(filename)


        # Read the data from the file. We first read the full file in an array
        # of bytes, to speed up I/O.
        buffer = np.fromfile(filename, dtype=np.byte)

        # Get the values from the buffer. They are continuous values of
        # (ID, SED_FILE, SED_POS, PDZ_FILE, PDZ_POS)
        self.__id_list = []
        self.__sed_file_list = []
        self.__sed_pos_list = []
        self.__pdz_file_list = []
        self.__pdz_pos_list = []
        file_size = os.path.getsize(filename)
        buf_pos = 0
        while buf_pos < file_size:
            self.__id_list.append(np.frombuffer(buffer, count=1, offset=buf_pos, dtype=np.int64)[0])
            buf_pos += 8
            self.__sed_file_list.append(np.frombuffer(buffer, count=1, offset=buf_pos, dtype=np.uint16)[0])
            buf_pos += 2
            self.__sed_pos_list.append(np.frombuffer(buffer, count=1, offset=buf_pos, dtype=np.int64)[0])
            buf_pos += 8
            self.__pdz_file_list.append(np.frombuffer(buffer, count=1, offset=buf_pos, dtype=np.uint16)[0])
            buf_pos += 2
            self.__pdz_pos_list.append(np.frombuffer(buffer, count=1, offset=buf_pos, dtype=np.int64)[0])
            buf_pos += 8

        # Check that the SED and PDZ positions are valid
        if np.any(np.asarray(self.__sed_pos_list) < -1):
            raise InvalidPositionException('Negative SED position values')
        if np.any(np.asarray(self.__pdz_pos_list) < -1):
            raise InvalidPositionException('Negative PDZ position values')

        # Create a map for easier search and at the same time check if we have
        # duplicates. The map values are (i, sed_file, sed_pos, pdz_file, pdz_pos),
        # where i is the index of the ID.
        self.__index_map = {}
        for i, (obj_id, sed_file, sed_pos, pdz_file, pdz_pos) in enumerate(zip(
                                        self.__id_list, self.__sed_file_list, self.__sed_pos_list,
                                        self.__pdz_file_list, self.__pdz_pos_list)):
            if obj_id in self.__index_map:
                raise IdMismatchException('Duplicate ID ' + str(obj_id))
            self.__index_map[obj_id] = [i, sed_file, sed_pos, pdz_file, pdz_pos]


    def size(self):
        """Returns the number of objects in the index"""
        return len(self.__id_list)


    def getIdList(self):
        """Returns a list of long integers with the IDs"""
        return self.__id_list


    def getSedFileList(self):
        """Returns a list of short unsigned integers with the SED file indices"""
        return self.__sed_file_list


    def getSedPositionList(self):
        """Returns a list of long integers with the positions in the SED data file"""
        return self.__sed_pos_list


    def getPdzFileList(self):
        """Returns a list of short unsigned integers with the PDZ file indices"""
        return self.__pdz_file_list


    def getPdzPositionList(self):
        """Returns a list of long integers with the positions in the PDZs data file"""
        return self.__pdz_pos_list


    def getFilesAndPositions(self, obj_id):
        """Returns the SED and PDZ file indices and positions for a given ID.

        Args:
            obj_id: The ID of the object to retrieve the info for

        Returns:
            A tuple with the following:
                - The SED file index
                - The SED position
                - The PDZ file index
                - The PDZ position

        Raises:
            IdMismatchException: If there is no such ID in the index
        """
        if not obj_id in self.__index_map:
            raise IdMismatchException('Index does not contain ID ' + str(obj_id))
        obj_info = self.__index_map[obj_id]
        return (obj_info[1], obj_info[2], obj_info[3], obj_info[4])


    def appendId(self, new_id):
        """Appends the given ID in the index, with 0 as the SED and PDZ file
         indices and -1 positions.

        Args:
            new_id: The new ID to append

        Raises:
            DuplicateIdException: If the ID is already in the index
        """

        # Check if we already have the ID in the index
        if new_id in self.__index_map:
            raise DuplicateIdException()

        # Write the values in the end of the file
        with open(self.__filename, 'ab') as f:
            np.asarray([new_id], dtype=np.int64).tofile(f)
            np.asarray([0], dtype=np.uint16).tofile(f) # SED file
            np.asarray([-1], dtype=np.int64).tofile(f) # SED position
            np.asarray([0], dtype=np.uint16).tofile(f) # PHZ file
            np.asarray([-1], dtype=np.int64).tofile(f) # PHZ position

        # Update the lists and the map
        self.__id_list.append(new_id)
        self.__sed_file_list.append(0)
        self.__sed_pos_list.append(-1)
        self.__pdz_file_list.append(0)
        self.__pdz_pos_list.append(-1)
        self.__index_map[new_id] = [len(self.__id_list)-1, 0, -1, 0, -1]


    def setSedFileAndPosition(self, obj_id, file_index, pos):
        """Sets the SED data file and position for the given object.

        Args:
            obj_id: The ID of the object to update the position for
            file_index: The index of the SED data file
            pos: The new position in the SED data file

        Raises:
            IdMismatchException: If the given ID is not in the index
            AlreadySetException: If the position is already se
        """

        # Check that the ID is managed by the index
        if not obj_id in self.__index_map:
            raise IdMismatchException('Index does not contain ID {}'.format(obj_id))
        i, _, old_sed_pos, _, _ = self.__index_map[obj_id]

        # Check that the position is not yet set
        if old_sed_pos != -1:
            raise AlreadySetException('SED position for ID {} is already set to {}'.format(obj_id, old_sed_pos))

        # Update the info in the index file
        with open(self.__filename, 'rb+') as f:
            f.seek(i * 28 + 8) # Skip the ID
            np.asarray([file_index], dtype=np.int16).tofile(f) # Write the file index
            np.asarray([pos], dtype=np.int64).tofile(f) # Write the position

        #Update the list and the map
        self.__sed_file_list[i] = file_index
        self.__sed_pos_list[i] = pos
        self.__index_map[obj_id][1] = file_index
        self.__index_map[obj_id][2] = pos


    def setPdzFileAndPosition(self, obj_id, file_index, pos):
        """Sets the PDZ data file and position for the given object.

        Args:
            obj_id: The ID of the object to update the position for
            file_index: The index of the PDZ data file
            pos: The new position in the PDZ data file

        Raises:
            IdMismatchException: If the given ID is not in the index
            AlreadySetException: If the position is already set
        """

        # Check that the ID is managed by the index
        if not obj_id in self.__index_map:
            raise IdMismatchException('Index does not contain ID {}'.format(obj_id))
        i, _, _, _, old_pdz_pos = self.__index_map[obj_id]

        # Check that the position is not yet set
        if old_pdz_pos != -1:
            raise AlreadySetException('PDZ position for ID {} is already set to {}'.format(obj_id, old_pdz_pos))

        # Update the info in the index file
        with open(self.__filename, 'rb+') as f:
            f.seek(i * 28 + 18)  # Skip the ID, SED file and SED position
            np.asarray([file_index], dtype=np.int16).tofile(f) # Write the file index
            np.asarray([pos], dtype=np.int64).tofile(f) # Write the position

        #Update the list and the map
        self.__pdz_file_list[i] = file_index
        self.__pdz_pos_list[i] = pos
        self.__index_map[obj_id][3] = file_index
        self.__index_map[obj_id][4] = pos


    def missingSedList(self):
        """Returns a list with the IDs of the objects for which the SED data have not been set"""
        result = []
        for obj_id, pos in zip(self.__id_list, self.__sed_pos_list):
            if pos == -1:
                result.append(obj_id)
        return result


    def missingPdzList(self):
        """Returns a list with the IDs of the objects for which the PDZ data have not been set"""
        result = []
        for obj_id, pos in zip(self.__id_list, self.__pdz_pos_list):
            if pos == -1:
                result.append(obj_id)
        return result
