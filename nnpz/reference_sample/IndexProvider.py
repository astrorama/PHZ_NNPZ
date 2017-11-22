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
            CorruptedFileException: If the size of the file is not divisible by 3
            IdMismatchException: If there are duplicate IDs
            InvalidPositionException: If SED or PDZ positions are not strictly
                increasing until the first -1 value
            InvalidPositionException: If a position is less than -1
            InvalidPositionException: If a positive position is found after a -1
        """
        self.__filename = filename
        
        if not os.path.exists(filename):
            raise FileNotFoundException(filename)
        
        # Check that the file size is consistent. It must contain triples of 8 byte
        # integers (24 bytes in total)
        if not os.path.getsize(filename) % 24 == 0:
            raise CorruptedFileException(filename)
        
        # Read the data from the file. They are continuous values of (ID, SED_POS, PDZ_POS)
        all_data = np.fromfile(filename, dtype=np.int64)
        all_data = all_data.reshape((3, len(all_data) // 3), order='F')
        self.__id_list = list(all_data[0,:])
        self.__sed_pos_list = list(all_data[1,:])
        self.__pdz_pos_list = list(all_data[2,:])
        
        # Check if the positions are valid
        self.__checkPositionValidity(self.__sed_pos_list, 'SED')
        self.__checkPositionValidity(self.__pdz_pos_list, 'PDZ')
        
        # Create a map for easier search and at the same time check if we have
        # duplicates. The map values are (i, sed_pos, pdz_pos), where i is the
        # index of the ID.
        self.__index_map = {}
        for i, (obj_id, sed_pos, pdz_pos) in enumerate(zip(self.__id_list, self.__sed_pos_list, self.__pdz_pos_list)):
            if obj_id in self.__index_map:
                raise IdMismatchException('Duplicate ID ' + str(obj_id))
            self.__index_map[obj_id] = [i, sed_pos, pdz_pos]
            
        # Keep track of the position of the first objects with SED and PDZ data set to -1
        self.__first_missing_sed_index = None
        for i, pos in enumerate(self.__sed_pos_list):
            if pos == -1:
                self.__first_missing_sed_index = i
                break
        self.__first_missing_pdz_index = None
        for i, pos in enumerate(self.__pdz_pos_list):
            if pos == -1:
                self.__first_missing_pdz_index = i
                break
            
            
    def size(self):
        """Returns the number of objects in the index"""
        return len(self.__id_list)
    
    
    def getIdList(self):
        """Returns a 1D numpy array of long integers with the IDs"""
        return self.__id_list
    
    
    def getSedPositionList(self):
        """Returns a 1D numpy array of long integers with the positions in the SED data file"""
        return self.__sed_pos_list
    
    
    def getPdzPositionList(self):
        """Returns a 1D numpy array of long integers with the positions in the PDZs data file"""
        return self.__pdz_pos_list
    
    
    def getPositions(self, obj_id):
        """Returns the SED and PDZ positions for a given ID.
        
        Args:
            obj_id: The ID of the object to retrieve the positions for
            
        Returns:
            A tuple with two values, the first being the SED position and the
            second the PDZ position
        
        Raises:
            IdMismatchException: If there is no such ID in the index
        """
        if not obj_id in self.__index_map:
            raise IdMismatchException('Index does not contain ID ' + str(obj_id))
        obj_info = self.__index_map[obj_id]
        return (obj_info[1], obj_info[2])
    
    
    def appendId(self, new_id):
        """Appends the given ID in the index, with -1 positions for the SED and PDZ files.
        
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
            np.asarray([new_id, -1, -1], dtype=np.int64).tofile(f)
        
        # Update the lists and the map
        self.__id_list.append(new_id)
        self.__sed_pos_list.append(-1)
        self.__pdz_pos_list.append(-1)
        self.__index_map[new_id] = [len(self.__id_list)-1, -1, -1]
        
        # Check if the newly created entry is the first with data set to -1
        if self.__first_missing_sed_index is None:
            self.__first_missing_sed_index = len(self.__id_list) - 1
        if self.__first_missing_pdz_index is None:
            self.__first_missing_pdz_index = len(self.__id_list) - 1
            
            
    def setSedPosition(self, obj_id, new_pos):
        """Sets the SED data file position for the given object.
        
        Args:
            obj_id: The ID of the object to update the position for
            pos: The new position in the SED data file
            
        Raises:
            IdMismatchException: If the given ID is not in the index
            AlreadySetException: If the position is already set
            InvalidPositionException: If the given position is less than the
                position of the previous object or the position of the previous
                object is not set
        """
        
        # Check that the ID is managed by the index
        if not obj_id in self.__index_map:
            raise IdMismatchException('Index does not contain ID ' + str(obj_id))
        i, old_sed_pos, _ = self.__index_map[obj_id]
        
        # Check that the position is not yet set
        if old_sed_pos != -1:
            raise AlreadySetException('SED position for ID ' + str(obj_id) + ' is already set to ' + str(old_sed_pos))
        
        # Check that the previous position is not -1 or greater
        if i != 0:
            if self.__sed_pos_list[i-1] == -1:
                raise InvalidPositionException('SED for previous ID (' + str(self.__id_list[i-1]) + ') is not set')
            if self.__sed_pos_list[i-1] >= new_pos:
                raise InvalidPositionException('New SED position is smaller than the previously stored')
            
        # Update the position in the file
        with open(self.__filename, 'rb+') as f:
            f.seek((i * 3 + 1) * 8)
            np.asarray([new_pos], dtype=np.int64).tofile(f)
        
        #Update the list and the map
        self.__sed_pos_list[i] = new_pos
        self.__index_map[obj_id][1] = new_pos
        
        # Update the index of the first missing SED object
        if i + 1 < len(self.__id_list):
            self.__first_missing_sed_index = i + 1
        else:
            self.__first_missing_sed_index = None
    
    
    def setPdzPosition(self, obj_id, new_pos):
        """Sets the PDZ data file position for the given object.
        
        Args:
            obj_id: The ID of the object to update the position for
            pos: The new position in the PDZ data file
            
        Raises:
            IdMismatchException: If the given ID is not in the index
            AlreadySetException: If the position is already set
            InvalidPositionException: If the given position is less or equal to
                the position of the previous object or the position of the
                previous object is not set
        """
        
        # Check that the ID is managed by the index
        if not obj_id in self.__index_map:
            raise IdMismatchException('Index does not contain ID ' + str(obj_id))
        i, _, old_pdz_pos = self.__index_map[obj_id]
        
        # Check that the position is not yet set
        if old_pdz_pos != -1:
            raise AlreadySetException('PDZ position for ID ' + str(obj_id) + ' is already set to ' + str(old_pdz_pos))
        
        # Check that the previous position is not -1 or greater
        if i != 0:
            if self.__pdz_pos_list[i-1] == -1:
                raise InvalidPositionException('PDZ for previous ID (' + str(self.__id_list[i-1]) + ') is not set')
            if self.__pdz_pos_list[i-1] >= new_pos:
                raise InvalidPositionException('New PDZ position is smaller than the previously stored')
            
        # Update the position in the file
        with open(self.__filename, 'rb+') as f:
            f.seek((i * 3 + 2) * 8)
            np.asarray([new_pos], dtype=np.int64).tofile(f)
        
        #Update the list and the map
        self.__pdz_pos_list[i] = new_pos
        self.__index_map[obj_id][2] = new_pos
        
        # Update the index of the first missing PDZ object
        if i + 1 < len(self.__id_list):
            self.__first_missing_pdz_index = i + 1
        else:
            self.__first_missing_pdz_index = None
        
        
    def firstIdMissingSedData(self):
        """Returns the ID of the first reference sample object which does not
        have SED data already set, or None if all SEDs are set"""
        if not self.__first_missing_sed_index is None:
            return self.__id_list[self.__first_missing_sed_index]
        else:
            return None
        
        
    def firstIdMissingPdzData(self):
        """Returns the ID of the first reference sample object which does not
        have PDZ data already set, or None if all PDZs are set"""
        if not self.__first_missing_pdz_index is None:
            return self.__id_list[self.__first_missing_pdz_index]
        else:
            return None