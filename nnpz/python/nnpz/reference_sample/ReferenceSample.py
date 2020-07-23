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
Created on: 09/11/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import os

import numpy as np
from ElementsKernel import Logging
from nnpz.exceptions import FileNotFoundException, CorruptedFileException, AlreadySetException, \
    InvalidDimensionsException, InvalidAxisException
from nnpz.reference_sample import IndexProvider, PdzDataProvider, SedDataProvider

logger = Logging.getLogger('ReferenceSample')


class ReferenceSample(object):
    """Object for handling the reference sample format of NNPZ"""

    default_index_filename = 'index.bin'
    sed_default_pattern = 'sed_data_{}.bin'
    pdz_default_pattern = 'pdz_data_{}.bin'

    @staticmethod
    def createNew(path, index_filename=None, sed_pattern=None, pdz_pattern=None):
        """Creates a new reference sample directory.

        Args:
            path:
                The path to create the reference sample in
            index_filename:
                The name of the index file. Defaults to `index.bin`
            sed_pattern:
                The pattern of the SED files. Defaults to `sed_data_{}.bin`
            pdz_pattern:
                The pattern of the PDZ files. Defaults to `pdz_data_{}.bin`

        Returns:
            An instance of the ReferenceSample class representing the new sample

        The newly created reference sample directory will contain an empty index
        file, an empty SED data file and an empty PDZ data file.
        """
        if index_filename is None:
            index_filename = ReferenceSample.default_index_filename
        if sed_pattern is None:
            sed_pattern = ReferenceSample.sed_default_pattern
        if pdz_pattern is None:
            pdz_pattern = ReferenceSample.pdz_default_pattern

        logger.debug('Creating reference sample directory ' + path + '...')

        # Create the directory
        os.makedirs(path)

        # Create an empty index file
        open(os.path.join(path, index_filename), 'wb').close()
        open(os.path.join(path, sed_pattern.format(1)), 'wb').close()
        open(os.path.join(path, pdz_pattern.format(1)), 'wb').close()

        return ReferenceSample(path, index_filename, sed_pattern, pdz_pattern)

    def __locate_existing_data_files(self, pattern):
        """Returns a set with the indices of the existing data files following the pattern"""
        result = set()
        i = 1
        while os.path.exists(pattern.format(i)):
            result.add(i)
            i += 1
        return result

    def __init__(self, path, index_filename=None, sed_pattern=None, pdz_pattern=None):
        """Creates a new ReferenceSample instance, managing the given path.

        Args:
            path:
                The path of the reference sample
            index_filename:
                The name of the index file. Defaults to `index.bin`
            sed_pattern:
                The pattern of the SED files. Defaults to `sed_data_{}.bin`
            pdz_pattern:
                The pattern of the PDZ files. Defaults to `pdz_data_{}.bin`

        Raises:
            FileNotFoundException: If any of the index or data files are not
                present in the given directory directory
        """
        if index_filename is None:
            index_filename = ReferenceSample.default_index_filename
        if sed_pattern is None:
            sed_pattern = ReferenceSample.sed_default_pattern
        if pdz_pattern is None:
            pdz_pattern = ReferenceSample.pdz_default_pattern

        # The file size which triggers the creation of a new data file
        self.__data_file_limit = 2**30  # 1GB

        # Construct the paths to all files
        self.__root_path = path
        self.__index_path = os.path.join(self.__root_path, index_filename)
        self.__sed_path_pattern = os.path.join(self.__root_path, sed_pattern)
        self.__pdz_path_pattern = os.path.join(self.__root_path, pdz_pattern)

        # Check that the directory and the index file exist
        if not os.path.exists(self.__root_path):
            raise FileNotFoundException(self.__root_path + ' does not exist')
        if not os.path.exists(self.__index_path):
            raise FileNotFoundException(self.__index_path + ' does not exist')

        # Initialize the internal handler for the index
        self.__index = IndexProvider(self.__index_path)

        # Check that all the SED files referred in the index exist
        existing_sed_files = self.__locate_existing_data_files(self.__sed_path_pattern)
        index_sed_files = set(self.__index.getSedFileList())
        index_sed_files.discard(0)  # We remove the zero, which means no file
        if not existing_sed_files.issuperset(index_sed_files):
            missing_sed = index_sed_files.difference(existing_sed_files)
            missing_files = list(map(self.__sed_path_pattern.format, missing_sed))
            raise FileNotFoundException(
                'Missing SED data files: {}'.format(', '.join(missing_files))
            )

        # Go through the SED files and create handlers
        self.__sed_map = {}
        for sed_file in existing_sed_files:
            self.__sed_map[sed_file] = SedDataProvider(self.__sed_path_pattern.format(sed_file))

        # Check that all the PDZ files referred in the index exist
        existing_pdz_files = self.__locate_existing_data_files(self.__pdz_path_pattern)
        index_pdz_files = set(self.__index.getPdzFileList())
        index_pdz_files.discard(0) # We remove the zero, which means no file
        if not existing_pdz_files.issuperset(index_pdz_files):
            missing_pdz = index_pdz_files.difference(existing_pdz_files)
            missing_files = list(map(self.__pdz_path_pattern.format, missing_pdz))
            raise FileNotFoundException(
                'Missing PDZ data files: {}'.format(', '.join(missing_files))
            )

        # Go through the PDZ files and create handlers
        self.__pdz_map = {}
        for pdz_file in existing_pdz_files:
            self.__pdz_map[pdz_file] = PdzDataProvider(self.__pdz_path_pattern.format(pdz_file))

    def size(self):
        """Returns the number of objects in the reference sample"""
        return self.__index.size()

    def getIds(self):
        """Returns the IDs of the reference sample objects as a numpy array of
        double precision (8 bytes) integers"""
        return self.__index.getIdList()

    def getSedData(self, obj_id):
        """Returns the SED data for the given reference sample object.

        Args:
            obj_id: The ID of the object to retrieve the SED for

        Returns:
            None if the SED is not set for the given object, otherwise the data
            of the SED as a two dimensional numpy array of single precision
            floats. The first dimension has size same as the number of the knots
            and the second dimension has always size equal to two, with the
            first element representing the wavelength and the second the energy
            value.

        Raises:
            IdMismatchException: If there is no such ID in the reference sample
            CorruptedFileException: If the ID stored in the index file is
                different than the one stored in the SED data file
        """

        # Get from the index the SED data file and the position in it
        sed_file, sed_pos, _, _ = self.__index.getFilesAndPositions(obj_id)

        # If it is not set yet, return None
        if sed_pos == -1:
            return None

        # Read the data from the SED data file
        file_id, sed_data = self.__sed_map[sed_file].readSed(sed_pos)

        # Check that the index and the SED data file are consistent
        if file_id != obj_id:
            raise CorruptedFileException(
                'Corrupted files: Index file contains the ID {} and SED data file the {}'.format(
                    obj_id, file_id)
            )

        return sed_data


    def getPdzData(self, obj_id):
        """Returns the PDZ data for the given reference sample object.

        Args:
            obj_id: The ID of the object to retrieve the PDZ for

        Returns:
            None if the PDZ is not set for the given object, otherwise the data
            of the PDZ as a two dimensional numpy array of single precision
            floats. The first dimension has size same as the number of the knots
            and the second dimension has always size equal to two, with the
            first element representing the wavelength and the second the
            probability value.

        Raises:
            IdMismatchException: If there is no such ID in the reference sample
            CorruptedFileException: If the ID stored in the index file is
                different than the one stored in the PDZ data file
        """

        # Get from the index the PDZ data file and the position in it
        _, _, pdz_file, pdz_pos = self.__index.getFilesAndPositions(obj_id)

        # If it is not set yet, return None
        if pdz_pos == -1:
            return None

        # Get the redshift bins of the PDZ. Note that this should never return
        # None if the pdz_pos is not -1.
        z_bins = self.__pdz_map[pdz_file].getRedshiftBins()

        # Read the data from the SED data file
        file_id, pdz_data = self.__pdz_map[pdz_file].readPdz(pdz_pos)

        # Check that the index and the PDZ data file are consistent
        if file_id != obj_id:
            raise CorruptedFileException(
                'Corrupted files: Index file contains the ID {} and PDZ data file the ID {}'.format(
                    obj_id, file_id)
            )

        result = np.ndarray((len(z_bins), 2), dtype=np.float32)
        result[:, 0] = z_bins
        result[:, 1] = pdz_data
        return result

    def createObject(self, new_id):
        """Creates a new object in the reference sample.

        Args:
            new_id: The ID of the object to create. It must be an integer.

        Throws:
            DuplicateIdException: if there is already an object with this ID in
                the reference sample

        When this method is called a new object in the reference sample is
        created. No SED or PDZ data are associated with this object. They can be
        set using the addSedData() and addPdzData() methods.
        """
        self.__index.appendId(new_id)

    def addSedData(self, obj_id, data):
        """Adds the SED data of a reference sample object.

        Args:
            obj_id: The ID of the object to add the SED for. It must be an integer.
            data: The data of the SED as a two dimensional array-like object.
                The first dimension has size same as the number of the knots and
                the second dimension has always size equal to two, with the
                first element representing the wavelength and the second the
                energy value.

        Raises:
            IdMismatchException: If the given ID is not in the reference sample
            AlreadySetException: If the SED data are already set for the given ID
            InvalidDimensionsException: If the given data dimensions are wrong
            InvalidAxisException: If there are decreasing wavelength values

        Note that if the latest SED data file size is bigger than 1GB, this
        method will result to the creation of a new SED data file.
        """

        # Check that the SED is not already set
        _, sed_pos, _, _ = self.__index.getFilesAndPositions(obj_id)
        if sed_pos != -1:
            raise AlreadySetException('SED for ID ' + str(obj_id) + ' is already set')

        # Add the SED data in the last file
        last_sed_file = max(self.__sed_map)
        new_pos = self.__sed_map[last_sed_file].appendSed(obj_id, data)
        self.__index.setSedFileAndPosition(obj_id, last_sed_file, new_pos)

        # Check if the last file exceeded the size limit and create a new one
        if os.path.getsize(self.__sed_path_pattern.format(last_sed_file)) >= self.__data_file_limit:
            new_sed_file = last_sed_file + 1
            filename = self.__sed_path_pattern.format(new_sed_file)
            open(filename, 'wb').close()
            self.__sed_map[new_sed_file] = SedDataProvider(filename)

    def missingSedList(self):
        """Returns a list with the IDs of the objects for which the SED data have
        not been set"""
        return self.__index.missingSedList()

    def addPdzData(self, obj_id, data):
        """Adds the PDZ data of a reference sample object.

        Args:
            obj_id: The ID of the object to add the PDZ for. It must be an integer.
            data: The data of the PDZ as a two dimensional array-like object.
                The first dimension has size same as the number of the knots and
                the second dimension has always size equal to two, with the
                first element representing the wavelength and the second the
                probability value.

        Raises:
            IdMismatchException: If the given ID is not in the reference sample
            AlreadySetException: If the PDZ data are aready set for the given ID
            InvalidDimensionsException: If the given data dimensions are wrong
            InvalidAxisException: If the wavelength values are not strictly increasing
            InvalidAxisException: If the wavelength values given are not matching
                the wavelength values of the other PDZs in the sample
        """

        # Check that the PDZ is not already set
        _, _, _, pdz_pos = self.__index.getFilesAndPositions(obj_id)
        if pdz_pos != -1:
            raise AlreadySetException('PDZ for ID ' + str(obj_id) + ' is already set')

        # Convert the data to a numpy array for easier handling
        data_arr = np.asarray(data, dtype=np.float32)
        if len(data_arr.shape) != 2 or data_arr.shape[1] != 2:
            raise InvalidDimensionsException()

        # Handle the wavelength axis
        last_pdz_file = max(self.__pdz_map)
        existing_zs = self.__pdz_map[last_pdz_file].getRedshiftBins()
        if existing_zs is None:
            self.__pdz_map[last_pdz_file].setRedshiftBins(data_arr[:, 0])
        elif not np.array_equal(data_arr[:, 0], existing_zs):
            raise InvalidAxisException('Given wavelengths are different than existing ones')

        # Add the PDZ data in the last file, normalizing first
        integral = np.trapz(data_arr[:, 1], data_arr[:, 0])
        new_pos = self.__pdz_map[last_pdz_file].appendPdz(obj_id, data_arr[:, 1] / integral)
        self.__index.setPdzFileAndPosition(obj_id, last_pdz_file, new_pos)

        # Check if the last file exceeded the size limit and create a new one
        if os.path.getsize(self.__pdz_path_pattern.format(last_pdz_file)) >= self.__data_file_limit:
            new_pdz_file = last_pdz_file + 1
            filename = self.__pdz_path_pattern.format(new_pdz_file)
            open(filename, 'wb').close()
            self.__pdz_map[new_pdz_file] = PdzDataProvider(filename)
            self.__pdz_map[new_pdz_file].setRedshiftBins(data_arr[:, 0])

    def missingPdzList(self):
        """Returns a list with the IDs of the objects for which the PDZ data have
        not been set"""
        return self.__index.missingPdzList()

    def iterate(self):
        """Returns an iterable object over the reference sample objects.

        The objects iterated provide the following members:
        - id: The ID of the object
        - sed: None if the SED is not set for the given object, otherwise the
            data of the SED as a two dimensional numpy array of single
            precission floats. The first dimension has size same as the number
            of the knots and the second dimension has always size equal to two,
            with the first element representing the wavelength and the second
            the energy value.
        - pdz: None if the PDZ is not set for the given object, otherwise the
            data of the PDZ as a two dimensional numpy array of single
            precission floats. The first dimension has size same as the number
            of the knots and the second dimension has always size equal to two,
            with the first element representing the wavelength and the second
            the probability value.
        """

        class Element(object):

            def __init__(self, obj_id, ref_sample):
                self.id = obj_id
                self.__ref_sample = ref_sample

            @property
            def sed(self):
                return self.__ref_sample.getSedData(self.id)

            @property
            def pdz(self):
                return self.__ref_sample.getPdzData(self.id)

        return (Element(i, self) for i in self.getIds())
