import os
from typing import Iterable

import numpy as np
from nnpz.exceptions import AlreadySetException
from nnpz.reference_sample import MontecarloDataProvider
from nnpz.reference_sample.BaseProvider import BaseProvider
from nnpz.reference_sample.util import validate_data_files, create_new_provider


class MontecarloProvider(BaseProvider):
    """
    A Montecarlo provider contains a set of N samples, each with D dimensions.
    Two examples would be:
        * Physical Parameters, where each dimension is a different physical parameter
        * Intermediate Bands for galaxies, where each dimension corresponds to the photometry for a
          given IB filter

    Args:
        *args:
            Forwarded to BaseProvider
        **kwargs:
            Forwarded to BaseProvider
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_files = validate_data_files(self._data_pattern, self._index, self._key)

        self._data_map = {}
        for data_file in data_files:
            self._data_map[data_file] = MontecarloDataProvider(
                self._data_pattern.format(data_file)
            )

    def flush(self):
        """
        Write the changes to disk.
        """
        self._index.flush()
        for data_prov in self._data_map.values():
            data_prov.flush()

    def getData(self, obj_id: int) -> np.ndarray:
        loc = self._index.get(obj_id, self._key)
        if not loc:
            return None
        return self._data_map[loc.file].read(loc.offset)

    def _getCurrentDataProvider(self):
        """
        Returns:
            A suitable data provider to add new data
        """
        if self._data_map:
            last_file = max(self._data_map)
        else:
            last_file = create_new_provider(
                self._data_map, self._data_pattern, MontecarloDataProvider
            )
        # Check if the last file exceeded the size limit and create a new one
        if self._data_map[last_file].size() >= self._data_limit:
            last_file = create_new_provider(
                self._data_map, self._data_pattern, MontecarloDataProvider
            )
        return last_file

    def addData(self, object_ids: Iterable[int] = None, data: np.ndarray = None):
        """
        Add new data to the MC provider

        Args:
            object_ids:
                Object ids
            data:
                New data
        """
        if len(data.shape) != 2:
            raise ValueError('Expecting an array with two axes')
        if len(object_ids) != data.shape[0]:
            raise ValueError('The number of objects and shape of the array do not match')

        record_size = data[0].nbytes
        records_per_file = self._data_limit // record_size + (self._data_limit % record_size > 0)

        # Index
        file_field = f'{self._key}_file'
        offset_field = f'{self._key}_offset'

        index_data = np.zeros(
            (len(object_ids),),
            dtype=[('id', np.int64), (file_field, np.int64), (offset_field, np.int64)]
        )

        # First available provider and merge whatever is possible
        provider_idx = self._getCurrentDataProvider()
        provider = self._data_map[provider_idx]
        available_size = self._data_limit - provider.size()
        nfit = available_size // record_size + (available_size % record_size > 0)

        index_data['id'][:nfit] = object_ids[:nfit]
        index_data[file_field][:nfit] = provider_idx
        index_data[offset_field][:nfit] = provider.append(data[:nfit])
        provider.flush()

        # Cut what's left in whole files
        n_files = int(np.ceil(len(object_ids[nfit:]) / records_per_file))

        for file_i in range(n_files):
            selection = slice(nfit + records_per_file * file_i,
                              nfit + records_per_file * (file_i + 1))

            provider_idx = self._getCurrentDataProvider()
            provider = self._data_map[provider_idx]
            index_data['id'][selection] = object_ids[selection]
            index_data[file_field][selection] = provider_idx
            index_data[offset_field][selection] = provider.append(data[selection])
            provider.flush()

        self._index.bulkAdd(index_data)
