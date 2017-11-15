"""
Created on: 10/11/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import os
import numpy as np

from nnpz.exceptions import *


class PdzDataProvider(object):
    """Class for handling the PDZ data file of NNPZ format"""
    
    
    def __init__(self, filename):
        """Creates a new instance for handling the given file.
        """
        self.__filename = filename
        # Read from the file the redshift beans, if the header is there
        self.__redshift_bins = None
        if os.path.getsize(self.__filename) > 0:
            with open(self.__filename, 'rb') as f:
                length = np.fromfile(f, count=1, dtype=np.uint32)[0]
                self.__redshift_bins = np.fromfile(f, count=length, dtype=np.float32)
        
        
    def setRedshiftBins(self, bins):
        """Sets the redshift bins of the PDZs in the file.
        
        Args:
            bins: The redshift bins values as a 1D numpy array of single
                precision floats
        
        Raises:
            InvalidDimensionsException: If the given bins array is not 1D
            InvalidAxisException: If the given bins values are not strictly
                increasing
            AlreadySetException: If the redhsift bins of the file are already set
        """
        
        if not self.__redshift_bins is None:
            raise AlreadySetException('PDZ redshift bins are already set')
        
        if len(bins.shape) != 1:
            raise InvalidDimensionsException('The bins array must be 1D')
        
        if (bins[:-1] >= bins[1:]).any():
            raise InvalidAxisException('Redshift bins must be strictly increasing')
        
        self.__redshift_bins = np.asarray(bins, dtype=np.float32)
        with open(self.__filename, 'ab') as f:
            np.asarray([len(self.__redshift_bins)], dtype=np.uint32).tofile(f)
            self.__redshift_bins.tofile(f)
        
        
    def getRedshiftBins(self):
        """Returns the redshift bins of the PDZs in the file.
        
        Returns: A 1D numpy array of single precision floats with the redshift
            bins or None if the redhsift bins are not set yet
        """
        return self.__redshift_bins
    
    
    def readPdz(self, pos):
        """Reads a PDZ from the given position.
        
        Args:
            pos: The position of the PDZ in the file
        
        Returns: A tuple with the following:
            - The ID of the PDZ
            - The data of the PDZ as a one dimensional numpy array of single
                precision floats.
        
        Raises:
            UninitializedException: If the redshift bins are not set
        """
        
        if self.__redshift_bins is None:
            raise UninitializedException()
        
        with open(self.__filename, 'rb') as f:
            f.seek(pos)
            pdz_id = np.fromfile(f, count=1, dtype=np.int64)[0]
            data = np.fromfile(f, count=len(self.__redshift_bins), dtype=np.float32)
        
        return pdz_id, data
    
    
    def appendPdz(self, pdz_id, data):
        """Appends a PDZ to the end of the file.
        
        Args:
            pdz_id: The ID of the PDZ to append
            data: The PDZ data as a single dimensional object
        
        Returns:
            Te position in the file where the PDZ was added
        
        Raises:
            InvalidDimensionsException: If the dimensions of the given data
                object are incorrect
            UninitializedException: If the redshift bins are not set
        """
        
        if self.__redshift_bins is None:
            raise UninitializedException()
        
        # Cnovert the data in a numpy array
        data_arr = np.asarray(data, dtype=np.float32)
        if len(data_arr.shape) != 1:
            raise InvalidDimensionsException('PDZ data must be a 1D array')
        if len(data_arr) != len(self.__redshift_bins):
            raise InvalidDimensionsException('PDZ data length differs from the redshift bins length')
        
        with open(self.__filename, 'ab') as f:
            pos = f.tell()
            np.asarray([pdz_id], dtype=np.int64).tofile(f)
            data_arr.tofile(f)
            
        return pos
        
        
    def validate(self, id_list, pos_list):
        """Validates that the underlying file is consistent with the given IDs and positions.
        
        Args:
            id_list: A list with the IDs of the reference sample objects
            pos_list: A list with the possitions of the PDZs in the file. All
                values after the first possition with value -1 are ignored.
            
        Returns: None if the file is consistent with the given IDs and positions,
            or a string message describing the first incossistency.
            
        Raises:
            InvalidDimensionsException: If the ID and possision lists have
                different length
            UninitializedException: If the redshift bins are not set
                
        The messages returned are for the following cases:
        - Incosistent ID
            The ID stored in the file for the given possition differs from the
            ID given by the user
            Message: Incosistent IDs (_USER_ID_, _FILE_ID_)
        - Incosistent position
            A position given by the user does not match the sum of the lengths
            of the previusly stored PDZs
            Message: Incosistent position for ID=_USER_ID_ (_USER_POS_, _FILE_POS_)
        - Exceeding filesize
            The expected length of the PDZs exceeds the file size
            Message: Expected data length bigger than file
        - Extra PDZs in file
            The file contains PDZs which are not in the given list
            Message: File contains extra PDZs
        """
        
        if self.__redshift_bins is None:
            raise UninitializedException()
        
        if len(id_list) != len(pos_list):
            raise InvalidDimensionsException('id_list and pos_list must have same length')
        
        # Trim the lists to the first -1 position
        try:
            i = pos_list.index(-1)
            id_list = id_list[:i]
            pos_list = pos_list[:i]
        except ValueError:
            pass # If there is no -1 in pos list do nothing
        
        expected_pos = 4 + 4 * len(self.__redshift_bins)
        pos_shift = 8 + 4 * len(self.__redshift_bins)
        
        # Check if it exceeds the file size
        file_size = os.path.getsize(self.__filename)
        if expected_pos + pos_shift * len(id_list) > file_size:
            return 'Expected data length bigger than file'
        
        with open(self.__filename, 'rb') as f:
            
            for pdz_id, pos in zip(id_list, pos_list):
                
                # Check that the pos is consistent
                if pos != expected_pos:
                    return 'Incosistent position for ID=' + str(pdz_id) + ' (' + str(pos) + ', ' + str(expected_pos) + ')'
                
                # Check that the ID is cosistent
                f.seek(pos)
                file_id = np.fromfile(f, count=1, dtype='int64')[0]
                if file_id != pdz_id:
                    return 'Incosistent IDs (' + str(pdz_id) + ', ' + str(file_id) + ')'
                
                # Update the expected_pos
                expected_pos = pos + pos_shift
        
        if expected_pos != file_size:
            return 'File contains extra PDZs'
        
        return None