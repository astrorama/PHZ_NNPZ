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
        """Creates a new instance for handling the given file.

        Raises:
            FileNotFoundException: If the file does not exist
        """
        self.__filename = filename

        if not os.path.exists(filename):
            raise FileNotFoundException(filename)


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
            InvalidAxisException: If there are decreasing wavelength values
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

        # Check that the wavelength axis does not have decreasing values
        if not np.all(data_arr[:-1,0] <= data_arr[1:,0]):
            raise InvalidAxisException('Wavelength axis must no have decreasing values')

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
        """Validates that the underlying file is consistent the given IDs and positions.

        Args:
            id_list: A list with the IDs of the reference sample objects
            pos_list: A list with the positions of the SEDs in the file.

        Returns: None if the file is consistent with the given IDs and
            positions, or a string message describing the first inconsistency.

        Raises:
            InvalidDimensionsException: If the ID and position lists have
                different length

        The messages returned are for the following cases:
        - Position out of file
            A given position is bigger than the file size
            Message: Position (_POS_) out of file for ID=_ID_
        - Inconsistent ID
            The ID stored in the file for the given position differs from the
            ID given by the user
            Message: Inconsistent IDs (_USER_ID_, _FILE_ID_)
        - Exceeding file size
            The length of a SED exceeds the file size
            Message: Data length bigger than file for ID=_USER_ID_
        """

        if len(id_list) != len(pos_list):
            raise InvalidDimensionsException('id_list and pos_list must have same length')

        # Check that all the given positions are smaller than the file size
        file_size = os.path.getsize(self.__filename)
        for id, pos in zip(id_list, pos_list):
            if pos >= file_size:
                return 'Position ({}) out of file for ID={}'.format(pos, id)

        with open(self.__filename, 'rb') as f:
            for id, pos in zip(id_list, pos_list):

                # Check that the ID given by the user are consistent with the file one
                f.seek(pos)
                file_id = np.fromfile(f, count=1, dtype='int64')[0]
                if file_id != id:
                    return 'Inconsistent IDs ({}, {})'.format(id, file_id)

                # Check that the length of the SED is not going out of the file
                length = np.fromfile(f, count=1, dtype=np.uint32)[0]
                sed_end = pos + 8 + 4 + (4 * 2 * length)
                if sed_end > file_size:
                    return 'Data length bigger than file for ID={}'.format(id)

        return None
