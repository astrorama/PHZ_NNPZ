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

import json
import os
import pathlib
from typing import Union, Iterable

import numpy as np
from ElementsKernel import Logging
from nnpz.exceptions import FileNotFoundException
from nnpz.reference_sample.MontecarloProvider import MontecarloProvider
from nnpz.reference_sample.PdzProvider import PdzProvider
from nnpz.reference_sample.SedProvider import SedProvider

logger = Logging.getLogger('ReferenceSample')


class ReferenceSample(object):
    """Object for handling the reference sample format of NNPZ"""

    PROVIDERS_FILE = 'providers.json'

    PROVIDER_MAP = dict(
        MontecarloProvider=MontecarloProvider,
        PdzProvider=PdzProvider,
        SedProvider=SedProvider
    )

    DEFAULT_PROVIDERS = {
        'PdzProvider': {'index': 'pdz_index.npy', 'data': 'pdz_data_{}.npy'},
        'SedProvider': {'index': 'sed_index.npy', 'data': 'sed_data_{}.npy'}
    }

    @staticmethod
    def createNew(path: Union[str, pathlib.Path], providers_file: str = PROVIDERS_FILE,
                  max_file_size=2 ** 30, providers=None):
        """
        Creates a new reference sample directory.

        Args:
            path:
                The path to create the reference sample in
            providers_file:
                The name of the providers file, a json with the list of data providers with their
                file names. It can be either absolute, or relative to `path`
            max_file_size:
                In bytes, the maximum size for data files
            providers: dict
                Set of provider configuration to initialize. Defaults to DEFAULT_PROVIDERS

        Returns:
            An instance of the ReferenceSample class representing the new sample
        """
        logger.debug('Creating reference sample directory %s...', path)
        os.makedirs(path)

        if providers is None:
            providers = ReferenceSample.DEFAULT_PROVIDERS

        with open(os.path.join(path, providers_file), 'wt') as fd:
            json.dump(dict(Providers=providers), fd, indent=2)

        return ReferenceSample(path, providers_file, max_file_size)

    def __setupProviders(self, providers_path):
        if not os.path.exists(providers_path):
            return {}

        with open(providers_path, 'rt') as fd:
            providers_json = json.load(fd)

        providers = dict()
        for provider_type_name, providers_config in providers_json.get('Providers', {}).items():
            logger.info('Found provider %s', provider_type_name)
            if not isinstance(providers_config, list):
                providers_config = [providers_config]

            for provider_config in providers_config:
                name = provider_config.get('name', provider_type_name)

                providers[name] = ReferenceSample.PROVIDER_MAP[provider_type_name](
                    os.path.join(self.__root_path, provider_config.pop('index')),
                    os.path.join(self.__root_path, provider_config.pop('data')),
                    self.__data_file_limit, provider_config
                )
        return providers

    def __init__(self, path: Union[str, pathlib.Path], providers_file: str = PROVIDERS_FILE,
                 max_file_size=2 ** 30):
        """Creates a new ReferenceSample instance, managing the given path.

        Args:
            path:
                The path to create the reference sample in
            providers_file:
                The name of the providers file, a json with the list of data providers with their
                file names. It can be either absolute, or relative to `path`
            max_file_size:
                In bytes, the maximum size for data files
        """
        self.__data_file_limit = max_file_size
        self.__root_path = path

        # Check that the directory and the index file exist
        if not os.path.exists(self.__root_path):
            raise FileNotFoundException(self.__root_path + ' does not exist')
        if not os.path.isdir(self.__root_path):
            raise NotADirectoryError(self.__root_path + ' is not a directory')

        self.__providers_path = os.path.join(path, providers_file)
        self.__providers = self.__setupProviders(self.__providers_path)

        if len(self.__providers):
            self.__all_ids = set(np.concatenate(
                [p.getIds() for p in self.__providers.values()]
            ))
        else:
            self.__all_ids = set()

    def __len__(self):
        """
        Returns the number of objects in the reference sample
        """
        return len(self.__all_ids)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()

    def _flush_config(self):
        providers_config = self._generateProviderConfig()
        with open(self.__providers_path, 'wt') as fd:
            json.dump(dict(Providers=providers_config), fd, indent=2)

    def flush(self):
        """
        Synchronize to disk
        """
        for prov in self.__providers.values():
            prov.flush()
        self._flush_config()

    def _generateProviderConfig(self):
        providers_config = {}
        for provider_name, provider in self.__providers.items():
            provider_type_name = type(provider).__name__
            if provider_type_name not in providers_config:
                providers_config[provider_type_name] = []
            config = dict(
                name=provider_name,
                index=os.path.basename(provider.index_path),
                data=os.path.basename(provider.data_pattern),
            )
            config.update(provider.extra)
            providers_config[provider_type_name].append(config)
        return providers_config

    def getIds(self) -> np.ndarray:
        """
        Returns the IDs of the reference sample objects as a numpy array of 64 bits integers
        """
        return self.__all_ids

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
        return self.__providers['SedProvider'].getSedData(obj_id)

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
        return self.__providers['PdzProvider'].getPdzData(obj_id)

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
        self.__all_ids.append(obj_id)
        return self.__providers['SedProvider'].addSedData(obj_id, data)

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
        self.__all_ids.append(obj_id)
        return self.__providers['PdzProvider'].addPdzData(obj_id, data)

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
        other_prov_path = os.path.join(other, os.path.basename(self.__providers_path))
        with open(other_prov_path, 'rt') as fd:
            other_providers_json = json.load(fd)

        for prov_type_name, provs_config in other_providers_json.get('Providers', {}).items():
            logger.info('Found provider %s', prov_type_name)
            if not isinstance(provs_config, list):
                provs_config = [provs_config]

            for prov_config in provs_config:
                name = prov_config.get('name', prov_type_name)
                index_name = prov_config.pop('index')
                data_pattern = prov_config.pop('data')
                if name not in self.__providers:
                    logger.info('Create provider %s of type %s', name, prov_type_name)
                    self.__providers[name] = ReferenceSample.PROVIDER_MAP[prov_type_name](
                        os.path.join(self.__root_path, index_name),
                        os.path.join(self.__root_path, data_pattern),
                        self.__data_file_limit, prov_config
                    )
                logger.info('Importing provider %s of type %s', name, prov_type_name)
                self.__providers[name].importData(
                    os.path.join(other, index_name),
                    os.path.join(other, data_pattern),
                    prov_config
                )

        # Update list of object IDs
        self.__all_ids = set(np.concatenate(
            [p.getIds() for p in self.__providers.values()]
        ))

    def addProvider(self, type_name, name=None, index_name: str = None, data_pattern: str = None,
                    object_ids: Iterable[int] = None, data: np.ndarray = None, extra: dict = None,
                    overwrite: bool = False):
        """

        Args:
            type_name:
            name:
            index_name:
            data_pattern:
            object_ids:
            data:
            extra:
            overwrite:

        Returns:

        """
        if name is None:
            name = type_name

        if not set(['name', 'index', 'data']).isdisjoint(extra.keys()):
            raise KeyError('Extra metadata can not be one of name, index or data')

        index_name = os.path.join(self.__root_path, index_name)
        if overwrite and os.path.exists(index_name):
            os.unlink(index_name)

        provider = self.PROVIDER_MAP[type_name](
            index_name, os.path.join(self.__root_path, data_pattern),
            self.__data_file_limit, extra)
        provider.initializeFromData(object_ids, data)
        provider.flush()
        self.__providers[name] = provider
        self._flush_config()
        self.__all_ids.update(object_ids)
