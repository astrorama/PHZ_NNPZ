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
from nnpz.exceptions import FileNotFoundException, AlreadySetException
from nnpz.reference_sample import IndexProvider
from nnpz.reference_sample.MontecarloProvider import MontecarloProvider
from nnpz.reference_sample.PdzProvider import PdzProvider
from nnpz.reference_sample.SedProvider import SedProvider

logger = Logging.getLogger('ReferenceSample')


class ReferenceSample(object):
    """Object for handling the reference sample format of NNPZ"""

    PROVIDER_MAP = dict(
        MontecarloProvider=MontecarloProvider,
        PdzProvider=PdzProvider,
        SedProvider=SedProvider
    )

    DEFAULT_PROVIDERS = {
        'PdzProvider': {'name': 'pdz', 'data': 'pdz_data_{}.npy'},
        'SedProvider': {'name': 'sed', 'data': 'sed_data_{}.npy'}
    }

    DEFAULT_INDEX = 'index.npy'

    @staticmethod
    def createNew(path: Union[str, pathlib.Path], index: str = None, providers: dict = None,
                  max_file_size=2 ** 30):
        """
        Creates a new reference sample directory.

        Args:
            path:
                The path to create the reference sample in
            index:
                The name of the index file. Defaults to DEFAULT_INDEX
            providers: dict
                Set of provider configuration to initialize. Defaults to DEFAULT_PROVIDERS
            max_file_size:
                In bytes, the maximum size for data files
        Returns:
            An instance of the ReferenceSample class representing the new sample
        """
        logger.debug('Creating reference sample directory %s...', path)
        os.makedirs(path)

        if not providers:
            providers = ReferenceSample.DEFAULT_PROVIDERS
        if not index:
            index = ReferenceSample.DEFAULT_INDEX

        return ReferenceSample(path, index, providers, max_file_size, create=True)

    def __setupProviders(self, providers_dict):
        providers = dict()
        for provider_type_name, providers_config in providers_dict.items():
            logger.info('Found provider %s', provider_type_name)
            if not isinstance(providers_config, list):
                providers_config = [providers_config]

            for provider_config in providers_config:
                config_copy = dict(provider_config)
                name = provider_config.get('name', provider_type_name)

                data_pattern = os.path.join(self.__root_path, config_copy.pop('data'))

                providers[name] = ReferenceSample.PROVIDER_MAP[provider_type_name](
                    self.__index, name, data_pattern, self.__data_file_limit, config_copy
                )
        return providers

    def __init__(self, path: Union[str, pathlib.Path], index: str = None, providers: dict = None,
                 max_file_size: int = 2 ** 30, create: bool = False):
        """Creates a new ReferenceSample instance, managing the given path.

        Args:
            path:
                The path to create the reference sample in
            index:
                The name of the index file. Defaults to DEFAULT_INDEX
            providers: dict
                Set of provider configuration to initialize. Defaults to DEFAULT_PROVIDERS
            max_file_size:
                In bytes, the maximum size for data files
            create:
                If True, fail if the required files exist, as this is supposed to be a new, empty,
                reference sample.
        """
        if not providers:
            providers = ReferenceSample.DEFAULT_PROVIDERS
        if not index:
            index = ReferenceSample.DEFAULT_INDEX

        self.__data_file_limit = max_file_size
        self.__root_path = path
        self.__index_path = os.path.join(self.__root_path, index)

        # Check that the directory exists
        if not os.path.exists(self.__root_path):
            raise FileNotFoundException(self.__root_path + ' does not exist')
        if not os.path.isdir(self.__root_path):
            raise NotADirectoryError(self.__root_path + ' is not a directory')
        if not create and not os.path.exists(self.__index_path):
            raise FileNotFoundException(self.__index_path + ' does not exist')

        self.__index = IndexProvider(self.__index_path)
        self.__providers = self.__setupProviders(providers)

    def __len__(self):
        """
        Returns the number of objects in the reference sample
        """
        return len(self.__index)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()

    def flush(self):
        """
        Synchronize to disk
        """
        for prov in self.__providers.values():
            prov.flush()
        self.__index.flush()

    def getIds(self) -> np.ndarray:
        """
        Returns the IDs of the reference sample objects as a numpy array of 64 bits integers
        """
        return self.__index.getIds()

    def getSedData(self, obj_id: int) -> Union[np.ndarray, None]:
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
        return self.__providers['sed'].getSedData(obj_id)

    def getPdzData(self, obj_id: int) -> Union[np.ndarray, None]:
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
        return self.__providers['pdz'].getPdzData(obj_id)

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
        return self.__providers['sed'].addSedData(obj_id, data)

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
        return self.__providers['pdz'].addPdzData(obj_id, data)

    def getData(self, name: str, obj_id: int):
        return self.__providers[name].getData(obj_id)

    def getProvider(self, name: str):
        return self.__providers[name]

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

        return (Element(i, self) for i in self.getIds())

    def importDirectory(self, other: Union[str, pathlib.Path]):
        """
        Bulk import of another reference sample

        Notes:
            The implementation only deals with files that are smaller than (or equal to) the
            limit for this reference sample.
        """
        other_index = IndexProvider(os.path.join(other, os.path.basename(self.__index_path)))
        for prov_name, provider in self.__providers.items():
            logger.info('Importing provider %s of type %s', prov_name, type(provider).__name__)
            provider.importData(
                other_index,
                os.path.join(other, os.path.basename(provider.data_pattern)),
                {}
            )

    def addProvider(self, type_name: str, name=None,
                    data_pattern: str = None, object_ids: Iterable[int] = None,
                    data: np.ndarray = None, extra: dict = None, overwrite: bool = False):
        """
        Register a new provider after the initialization of the reference sample
        Args:
            type_name: str
                Provider type name, one of PROVIDER_MAP
            name: str
                Provider unique name, defaults to type_name if not given.
            data_pattern: str
                Pattern for the data file names
            object_ids: iterable
                Initial set of object ids
            data: np.ndarray
                Initial data set
            extra: dict
                Any additional metadata
            overwrite: bool
                If True, the destination files will be removed if they exist
        """
        if name is None:
            name = type_name

        if extra and not set(['name', 'index', 'data']).isdisjoint(extra.keys()):
            raise KeyError('Extra metadata can not be one of name, index or data')

        if name in self.__providers and not overwrite:
            raise AlreadySetException(name)

        provider = self.PROVIDER_MAP[type_name](
            self.__index, name, os.path.join(self.__root_path, data_pattern),
            self.__data_file_limit, extra, overwrite=overwrite)

        if object_ids is not None and data is not None:
            provider.addData(object_ids, data)
            provider.flush()
        self.__providers[name] = provider
        return provider
