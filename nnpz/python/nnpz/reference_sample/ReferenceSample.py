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
import pathlib
from typing import Union, Iterable

import numpy as np
from ElementsKernel import Logging
from nnpz.exceptions import FileNotFoundException, AlreadySetException, \
    InvalidDimensionsException, InvalidAxisException
from nnpz.reference_sample import IndexProvider, PdzDataProvider, SedDataProvider

logger = Logging.getLogger('ReferenceSample')


class ReferenceSample(object):
    """Object for handling the reference sample format of NNPZ"""

    SED_DEFAULT_PATTERN = 'sed_data_{}.npy'
    SED_DEFAULT_INDEX = 'sed_index.npy'
    PDZ_DEFAULT_PATTERN = 'pdz_data_{}.npy'
    PDZ_DEFAULT_INDEX = 'pdz_index.npy'
    PP_DEFAULT_PATTERN = 'pp_data_{}.npy'
    PP_DEFAULT_INDEX = 'pp_index.npy'

    @staticmethod
    def createNew(path: Union[str, pathlib.Path], **kwargs):
        """
        Creates a new reference sample directory.

        Args:
            path:
                The path to create the reference sample in
            **kwargs:
                Forwarded to the constructor

        Returns:
            An instance of the ReferenceSample class representing the new sample
        """
        logger.debug('Creating reference sample directory %s...', path)
        os.makedirs(path)
        return ReferenceSample(path, **kwargs)

    @staticmethod
    def __locate_existing_data_files(pattern):
        """
        Returns a set with the indices of the existing data files following the pattern
        """
        result = set()
        i = 1
        while os.path.exists(pattern.format(i)):
            result.add(i)
            i += 1
        return result

    @staticmethod
    def __validate_data_files(pattern: str, index: IndexProvider, label: str):
        """
        Cross-check the existing data files and those referenced by the index
        """
        existing_files = ReferenceSample.__locate_existing_data_files(pattern)
        index_files = index.getFiles()
        if not existing_files.issuperset(index_files):
            missing_files = index_files.difference(existing_files)
            missing_files = list(map(pattern.format, missing_files))
            raise FileNotFoundException(
                'Missing {} data files: {}'.format(label, ', '.join(missing_files))
            )
        return existing_files

    @staticmethod
    def __create_new_provider(provider_map: dict, pattern: str, class_: type):
        """
        Instantiate a new provider
        """
        if provider_map:
            last = max(provider_map)
            provider_map[last].flush()
            new_file = last + 1
        else:
            new_file = 1
        filename = pattern.format(new_file)
        provider_map[new_file] = class_(filename)
        return new_file

    def __init__(self, path: Union[str, pathlib.Path],
                 sed_index: str = SED_DEFAULT_INDEX, sed_pattern: str = SED_DEFAULT_PATTERN,
                 pdz_index: str = PDZ_DEFAULT_INDEX, pdz_pattern: str = PDZ_DEFAULT_PATTERN,
                 pp_index: str = PP_DEFAULT_INDEX, pp_default_pattern: str = PP_DEFAULT_PATTERN,
                 max_file_size=2 ** 30):
        """Creates a new ReferenceSample instance, managing the given path.

        Args:
            path:
                The path to create the reference sample in
            sed_index:
                The name of the SED index file. Defaults to `sed_index.npy`
            sed_pattern:
                The pattern of the SED files. Defaults to `sed_data_{}.npy`
            pdz_index:
                The name of the PDZ index file. Defaults to `pdz_index.npy`
            pdz_pattern:
                The pattern of the PDZ files. Defaults to `pdz_data_{}.npy`
            pp_index:
                The name of the Physical Parameters index file. Defaults to `pp_index.npy`
            pp_default_pattern:
                The pattern of the Physical Parameters files. Defaults to `pp_data_{}.npy`
            max_file_size:
                In bytes, the maximum size for data files
        """
        # The file size which triggers the creation of a new data file
        self.__data_file_limit = max_file_size

        # Construct the paths to all files
        self.__root_path = path
        self.__sed_index_path = os.path.join(self.__root_path, sed_index)
        self.__sed_path_pattern = os.path.join(self.__root_path, sed_pattern)
        self.__pdz_index_path = os.path.join(self.__root_path, pdz_index)
        self.__pdz_path_pattern = os.path.join(self.__root_path, pdz_pattern)
        self.__pp_index_path = os.path.join(self.__root_path, pp_index)
        self.__pp_path_pattern = os.path.join(self.__root_path, pp_default_pattern)

        # Check that the directory and the index file exist
        if not os.path.exists(self.__root_path):
            raise FileNotFoundException(self.__root_path + ' does not exist')
        if not os.path.isdir(self.__root_path):
            raise NotADirectoryError(self.__root_path + ' is not a directory')

        # Initialize the internal handler for the index
        self.__sed_index = IndexProvider(self.__sed_index_path)
        self.__pdz_index = IndexProvider(self.__pdz_index_path)
        self.__pp_index = IndexProvider(self.__pp_index_path)

        # Check that all the SED files referred in the index exist
        sed_files = self.__validate_data_files(self.__sed_path_pattern, self.__sed_index, 'SED')

        # Go through the SED files and create handlers
        self.__sed_map = {}
        self.__sed_prov_for_size = {}
        for sed_file in sed_files:
            sed_prov = SedDataProvider(self.__sed_path_pattern.format(sed_file))
            self.__sed_map[sed_file] = sed_prov
            self.__sed_prov_for_size[sed_prov.getKnots()] = sed_file

        # Check that all the PDZ files referred in the index exist
        pdz_files = ReferenceSample.__validate_data_files(
            self.__pdz_path_pattern, self.__pdz_index, 'PDZ')

        # Go through the PDZ files and create handlers
        self.__pdz_map = {}
        for pdz_file in pdz_files:
            self.__pdz_map[pdz_file] = PdzDataProvider(self.__pdz_path_pattern.format(pdz_file))

    def __len__(self):
        """
        Returns the number of objects in the reference sample
        """
        return max(len(self.__sed_index), len(self.__pdz_index))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()

    def flush(self):
        """
        Synchronize to disk
        """
        for pdz_prov in self.__pdz_map.values():
            pdz_prov.flush()
        for sed_prov in self.__sed_map.values():
            sed_prov.flush()
        self.__pdz_index.flush()
        self.__sed_index.flush()

    def getIds(self) -> np.ndarray:
        """
        Returns the IDs of the reference sample objects as a numpy array of 64 bits integers
        """
        return np.unique(np.concatenate([
            self.__sed_index.getIds(), self.__pdz_index.getIds(), self.__pp_index.getIds()]
        ))

    def getSedData(self, obj_id: int) -> np.ndarray:
        """
        Returns the SED data for the given reference sample object.

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
        sed_loc = self.__sed_index.get(obj_id)
        if sed_loc:
            return self.__sed_map[sed_loc.file].readSed(sed_loc.offset)
        return None

    def getPdzData(self, obj_id: int) -> np.ndarray:
        """
        Returns the PDZ data for the given reference sample object.

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
        pdz_loc = self.__pdz_index.get(obj_id)
        if not pdz_loc:
            return None
        z_bins = self.__pdz_map[pdz_loc.file].getRedshiftBins().reshape(-1, 1)
        pdz_data = self.__pdz_map[pdz_loc.file].readPdz(pdz_loc.offset).reshape(-1, 1)
        return np.hstack([z_bins, pdz_data])

    def addSedData(self, obj_id: int, data: np.ndarray):
        """
        Adds the SED data of a reference sample object.

        Args:
            obj_id: The ID of the object to add the SED for. It must be an integer.
            data: The data of the SED as a two dimensional array-like object.
                The first dimension has size same as the number of the knots and
                the second dimension has always size equal to two, with the
                first element representing the wavelength and the second the
                energy value.

        Raises:
            AlreadySetException: If the SED data are already set for the given ID
            InvalidDimensionsException: If the given data dimensions are wrong
            InvalidAxisException: If there are decreasing wavelength values

        Note that if the latest SED data file size is bigger than 2GB, this
        method will result to the creation of a new SED data file.
        """
        # Check that the SED is not already set
        loc = self.__sed_index.get(obj_id)
        if loc is not None:
            raise AlreadySetException('SED for ID ' + str(obj_id) + ' is already set')

        current_prov = self._getCurrentSedProvider(data.shape[0])
        new_pos = self.__sed_map[current_prov].appendSed(data)
        self.__sed_index.add(obj_id, IndexProvider.ObjectLocation(current_prov, new_pos))

    def _getCurrentSedProvider(self, knots):
        """
        Get the index of the active SED provider, create a new one if needed
        """
        if knots not in self.__sed_prov_for_size \
            or self.__sed_map[self.__sed_prov_for_size[knots]].size() >= self.__data_file_limit:
            self.__sed_prov_for_size[knots] = self.__create_new_provider(
                self.__sed_map, self.__sed_path_pattern, SedDataProvider
            )
        return self.__sed_prov_for_size[knots]

    def addPdzData(self, obj_id, data):
        """
        Adds the PDZ data of a reference sample object.

        Args:
            obj_id: The ID of the object to add the PDZ for. It must be an integer.
            data: The data of the PDZ as a two dimensional array-like object.
                The first dimension has size same as the number of the knots and
                the second dimension has always size equal to two, with the
                first element representing the wavelength and the second the
                probability value.

        Raises:
            AlreadySetException: If the PDZ data are aready set for the given ID
            InvalidDimensionsException: If the given data dimensions are wrong
            InvalidAxisException: If the wavelength values are not strictly increasing
            InvalidAxisException: If the wavelength values given are not matching
                the wavelength values of the other PDZs in the sample
        """
        # Check that the PDZ is not already set
        loc = self.__pdz_index.get(obj_id)
        if loc is not None:
            raise AlreadySetException('PDZ for ID ' + str(obj_id) + ' is already set')

        # Convert the data to a numpy array for easier handling
        data_arr = np.asarray(data, dtype=np.float32)
        if len(data_arr.shape) != 2 or data_arr.shape[1] != 2:
            raise InvalidDimensionsException()

        pdz_file = self._getCurrentPdzProvider(data_arr[:, 0])

        # Add the PDZ data in the last file, normalizing first
        integral = np.trapz(data_arr[:, 1], data_arr[:, 0])
        new_pos = self.__pdz_map[pdz_file].appendPdz(data_arr[:, 1] / integral)
        self.__pdz_index.add(obj_id, IndexProvider.ObjectLocation(pdz_file, new_pos))

    def _getCurrentPdzProvider(self, binning):
        """
        Get the index of the active PDZ provider, create a new one if needed
        """
        if self.__pdz_map:
            last_pdz_file = max(self.__pdz_map)
        else:
            last_pdz_file = self.__create_new_provider(
                self.__pdz_map, self.__pdz_path_pattern, PdzDataProvider
            )

        # Check if the last file exceeded the size limit and create a new one
        if self.__pdz_map[last_pdz_file].size() >= self.__data_file_limit:
            last_pdz_file = self.__create_new_provider(
                self.__pdz_map, self.__pdz_path_pattern, PdzDataProvider
            )

        # Set, or crosscheck, the binning
        existing_zs = self.__pdz_map[last_pdz_file].getRedshiftBins()
        if existing_zs is None:
            self.__pdz_map[last_pdz_file].setRedshiftBins(binning)
        elif not np.array_equal(binning, existing_zs):
            raise InvalidAxisException('Given wavelengths are different than existing ones')

        return last_pdz_file

    def getPhysicalParameters(self, obj_id: int) -> np.ndarray:
        """
        Return the Physical Parameters data for the given reference sample object.

        Args:
            obj_id: The ID of the object to retrieve the data for

        Returns:
            None if the parameters are not set for the given object, otherwise the data
            as a two dimensional numpy array of single precision floats.
            The first dimension has size same as the number of samples,
            and the second dimension always size equal to the dimensionality of the space.
        """
        pp_loc = self.__pp_index.get(obj_id)
        if not pp_loc:
            return None
        return self.__pp_map[pp_loc.file].read(pp_loc.offset)

    def addPhysicalParameters(self, obj_id: int, data: np.ndarray):
        """
        Add a new entry of Physical Parameters
        Args:
            obj_id: The ID of the object to store the data for
            data

        Raises:
            AlreadySetException: If the data are already set for the given ID
        """
        # Check that the PDZ is not already set
        loc = self.__pp_index.get(obj_id)
        if loc is not None:
            raise AlreadySetException('Physical parameters for ID {} already set'.format(obj_id))

        pp_file = self._getCurrentPhysicalParameterProvider()
        offset = self.__pp_map[pp_file].append(data)
        self.__pp_index.add(obj_id, IndexProvider.ObjectLocation(pp_file, offset))

    def _getCurrentPhysicalParameterProvider(self):
        """
        Get the index of the active PP provider, create a new one if needed
        """
        if self.__pp_map:
            last_pp_file = max(self.__pp_map)
        else:
            last_pp_file = self.__create_new_provider(
                self.__pp_map, self.__pp_path_pattern, PhysicalParameterProvider
            )

        # Check if the last file exceeded the size limit and create a new one
        if self.__pp_map[last_pp_file].size() >= self.__data_file_limit:
            last_pp_file = self.__create_new_provider(
                self.__pp_map, self.__pp_path_pattern, PhysicalParameterProvider
            )

        return last_pp_file

    def iterate(self) -> Iterable:
        """
        Returns an iterable object over the reference sample objects.

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

            @property
            def physical_params(self):
                return self.__ref_sample.getPhysicalParameters(self.id)

        return (Element(i, self) for i in self.getIds())

    def importDirectory(self, other: Union[str, pathlib.Path]):
        """
        Bulk import of another reference sample

        Notes:
            The file patterns must be the same.
            The implementation only deals with files that are smaller than (or equal to) the
            limit for this reference sample.
        """
        pdz_pattern = os.path.join(other, os.path.basename(self.__pdz_path_pattern))
        pdz_index = np.load(
            os.path.join(other, os.path.basename(self.__pdz_index_path)),
            mmap_mode='r'
        )
        sed_pattern = os.path.join(other, os.path.basename(self.__sed_path_pattern))
        sed_index = np.load(
            os.path.join(other, os.path.basename(self.__sed_index_path)),
            mmap_mode='r'
        )

        self.importPdz(pdz_pattern, pdz_index)
        self.importSed(sed_pattern, sed_index)

    def importPdz(self, pdz_pattern: str, pdz_index: np.ndarray):
        """
        Import a set of PDZ files
        """
        pdz_files = sorted(self.__locate_existing_data_files(pdz_pattern))
        for pdz_i in pdz_files:
            pdz_file = pdz_pattern.format(pdz_i)
            pdz_size = os.path.getsize(pdz_file)
            pdz = np.load(pdz_file, mmap_mode='r')
            pdz_index_pos = np.nonzero(pdz_index[:, 1] == pdz_i)[0]
            updated_index = np.array(pdz_index[pdz_index_pos], copy=True)

            pdz_provider_idx = self._getCurrentPdzProvider(pdz[0, :])
            pdz_provider = self.__pdz_map[pdz_provider_idx]

            # Fit whatever we can on the current file (approximately)
            available_size = self.__data_file_limit - pdz_provider.size()
            nfit = len(pdz) * min(np.ceil(available_size / pdz_size), 1)
            updated_index[:nfit, 1] = pdz_provider_idx
            updated_index[:nfit, 2] = pdz_provider.appendPdz(pdz[1:nfit])

            # Create a new one and put in the rest
            pdz_provider_idx = self._getCurrentPdzProvider(pdz[0, :])
            pdz_provider = self.__pdz_map[pdz_provider_idx]
            updated_index[nfit:, 1] = pdz_provider_idx
            updated_index[nfit:, 2] = pdz_provider.appendPdz(pdz[nfit:])

            # Update the index
            self.__pdz_index.bulkAdd(updated_index)

    def importSed(self, sed_pattern: str, sed_index: np.ndarray):
        """
        Import a set of SED files
        """
        sed_files = sorted(self.__locate_existing_data_files(sed_pattern))
        for sed_i in sed_files:
            sed_file = sed_pattern.format(sed_i)
            sed_size = os.path.getsize(sed_file)
            sed = np.load(sed_file, mmap_mode='r')
            sed_index_pos = np.where(sed_index[:, 1] == sed_i)[0]
            updated_index = np.array(sed_index[sed_index_pos], copy=True)

            sed_provider_idx = self._getCurrentSedProvider(sed.shape[1])
            sed_provider = self.__sed_map[sed_provider_idx]

            # Fit whatever we can on the current file (approximately)
            available_size = self.__data_file_limit - sed_provider.size()
            nfit = len(sed) * min(np.ceil(available_size / sed_size), 1)
            updated_index[:nfit, 1] = sed_provider_idx
            updated_index[:nfit, 2] = sed_provider.appendSed(sed[:nfit])

            # Create a new one and put in the rest
            sed_provider_idx = self._getCurrentSedProvider(sed.shape[1])
            sed_provider = self.__sed_map[sed_provider_idx]
            updated_index[nfit:, 1] = sed_provider_idx
            updated_index[nfit:, 2] = sed_provider.appendSed(sed[nfit:])

            # Update the index
            self.__sed_index.bulkAdd(updated_index)
