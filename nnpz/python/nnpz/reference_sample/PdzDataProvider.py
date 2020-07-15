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


class PdzDataProvider(object):
    """Class for handling the PDZ data file of NNPZ format"""


    def __init__(self, filename):
        """Creates a new instance for handling the given file.

        Raises:
            FileNotFoundException: If the file does not exist
        """
        self.__filename = filename

        if not os.path.exists(filename):
            raise FileNotFoundException(filename)

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
            pos_list: A list with the positions of the PDZs in the file

        Returns: None if the file is consistent with the given IDs and positions,
            or a string message describing the first inconsistency.

        Raises:
            InvalidDimensionsException: If the ID and posision lists have
                different length
            UninitializedException: If the redshift bins are not set

        The messages returned are for the following cases:
        - Position out of file
            A given position is bigger than the file size
            Message: Position (_POS_) out of file for ID=_ID_
        - Inconsistent ID
            The ID stored in the file for the given position differs from the
            ID given by the user
            Message: Inconsistent IDs (_USER_ID_, _FILE_ID_)
        - Exceeding file size
            The length of a PDZ exceeds the file size
            Message: Data length bigger than file for ID=_USER_ID_
        """

        if self.__redshift_bins is None:
            raise UninitializedException()

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

                # Check that the length of the PDZ is not going out of the file
                pdz_end = pos + 8 + 4 * len(self.__redshift_bins)
                if pdz_end > file_size:
                    return 'Data length bigger than file for ID={}'.format(id)

        return None
