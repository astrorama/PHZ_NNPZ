"""
Created on: 10/11/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import os
import numpy as np

from nnpz.exceptions import *


class SedDataProvider(object):
    """Class for handling the SED data file of NNPZ format"""
    
    
    def __init__(self, filename):
        """Creates a new instance for handling the given file"""
        self.__filename = filename
    
    
    def readSed(self, pos):
        """Reads the data of an SED.
        
        Args:
            pos: The possition of the SED in the file
        
        Returns: A tuple with the following:
            - The ID of the SED
            - The data of the SED as a two dimensional numpy array of single
                precission floats. The first dimension has size same as the
                number of the knots and the second dimension has always size
                equal to two, with the first element representing the wavelength
                and the second the energy value.
        """
        
        with open(self.__filename, 'rb') as f:
            # Move where the SED is
            f.seek(pos)
            
            # Read he ID
            sed_id = np.fromfile(f, count=1, dtype='int64')[0]
            
            # Read the data
            length = np.fromfile(f, count=1, dtype='uint32')[0]
            data = np.fromfile(f, count=2*length, dtype='float32')
            data = data.reshape((length, 2))
            
            return sed_id, data
    
    
    def appendSed(self, sed_id, data):
        """Appends an SED to the end of the file.
        
        Args:
            sed_id: The ID of the SED to append
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
            InvalidAxisException: If the wavelength values of the SED are not strictly
                increasing.
        """
        
        # First convert the data in a numpy array for easier handling
        data_arr = np.asarray(data, dtype='float32')
        
        # Check the dimensions
        if len(data_arr.shape) != 2:
            raise InvalidDimensionsException('data must be a two dimensional array' +
                ' but it had ' + str(len(data_arr.shape)) + ' dimensions')
        if data_arr.shape[1] != 2:
            raise InvalidDimensionsException('data second dimension must be of size' +
                ' 2 but was ' + str(data_arr.shape[1]))
                
        # Check that the wavelength axis is striictly increasing
        if not np.all(data_arr[:-1,0] < data_arr[1:,0]):
            raise InvalidAxisException('Wavelength axis must be strictly increasing')
        
        # The position of the new data will be the current size of the file
        sed_pos = os.path.getsize(self.__filename)
        
        with open(self.__filename, 'ab') as f:
            # Store the ID as 8 byte long signed integer
            np.asarray([sed_id], dtype='int64').tofile(f)
            
            # Store the length of the data as 4 byte unsigned integer
            np.asarray([len(data)], dtype='uint32').tofile(f)
            
            # Store the data of the sed as 4 byte floats, as pairs of wavelength
            # flux values
            data_arr.flatten().tofile(f)
        
        return sed_pos
    
    
    def validate(self, id_list, pos_list):
        """Validates that the underlying file is consistent the given IDs and possitions.
        
        Args:
            id_list: A list with the IDs of the reference sample objects
            pos_list: A list with the possitions of the SEDs in the file
        
        Returns: None if the file is consistent with the given IDs and
            possitions, or a string message describing the first incossistency.
        
        Raises:
            InvalidDimensionsException: If the ID and possision lists have
                different length
                
        The messages returned are for the following cases:
        - Incosistent ID
            The ID stored in the file for the given possition differs from the
            ID given by the user
            Message: Incosistent IDs (_USER_ID_, _FILE_ID_)
        - Incosistent position
            A position given by the user does not match the sum of the lengths
            of the previusly stored SEDs
            Message: Incosistent position for ID=_USER_ID_ (_USER_POS_, _FILE_POS_)
        - Exceeding filesize
            The length of an SED exceeds the file size
            Message: Data length bigger than file for ID=_USER_ID_
        """
        
        if len(id_list) != len(pos_list):
            raise InvalidDimensionsException('id_list and pos_list must have same length')
        
        file_size = os.path.getsize(self.__filename)
        with open(self.__filename, 'rb') as f:
            
            expected_pos = 0
            for sed_id, pos in zip(id_list, pos_list):
                
                # Check that the pos is consistent
                if pos != expected_pos:
                    return 'Incosistent position for ID=' + str(sed_id) + ' (' + str(pos) + ', ' + str(expected_pos) + ')'
                
                # Check that the ID is cosistent
                f.seek(pos)
                file_id = np.fromfile(f, count=1, dtype='int64')[0]
                if file_id != sed_id:
                    return 'Incosistent IDs (' + str(sed_id) + ', ' + str(file_id) + ')'
                
                # Get the length from the file
                length = np.fromfile(f, count=1, dtype='uint32')[0]
                
                # Update the expected_pos
                expected_pos = pos + 8 + 4 + (4 * 2 * length)
                
                # Check it is less than the file size
                if expected_pos > file_size:
                    return 'Data length bigger than file for ID=' + str(sed_id)
        
        return None