import numpy as np
from nnpz.exceptions import UninitializedException
from nnpz.reference_sample import MontecarloDataProvider
from nnpz.reference_sample.BaseProvider import BaseProvider
from nnpz.reference_sample.util import validate_data_files
from typing import Iterable


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
        self._data_files = validate_data_files(self._data_pattern, self._index, self._key)
        self._current_data_provider = None
        self._current_data_index = None
        self._last_data_index = max(self._data_files) if self._data_files else None

    def flush(self):
        """
        Write the changes to disk.
        """
        self._index.flush()
        if self._current_data_provider:
            self._current_data_provider.flush()

    def _swapProvider(self, index):
        if index != self._current_data_index:
            if self._current_data_provider is not None:
                self._current_data_provider.flush()
            self._current_data_index = index
            self._current_data_provider = MontecarloDataProvider(
                self._data_pattern.format(index)
            )

    def getDtype(self, parameter):
        """
        Returns:
            The dtype of the given parameter
        """
        if not self._data_map:
            raise UninitializedException('MontecarloProvider not initialized')
        return next(iter(self._data_map.values())).read(0)[parameter].dtype

    def getData(self, obj_id: int) -> np.ndarray:
        loc = self._index.get(obj_id, self._key)
        if not loc:
            return None
        self._swapProvider(loc.file)
        return self._current_data_provider.read(loc.offset)

    def _getWriteableDataProvider(self):
        """
        Returns:
            A suitable data provider to add new data
        """
        if not self._last_data_index:
            self._last_data_index = 1
            self._swapProvider(self._last_data_index)

        # Check if the last file exceeded the size limit and create a new one
        if self._current_data_provider.size() >= self._data_limit:
            self._last_data_index += 1
            self._swapProvider(self._last_data_index)

        return self._last_data_index, self._current_data_provider

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
        provider_idx, provider = self._getWriteableDataProvider()
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

            provider_idx, provider = self._getWriteableDataProvider()
            index_data['id'][selection] = object_ids[selection]
            index_data[file_field][selection] = provider_idx
            index_data[offset_field][selection] = provider.append(data[selection])

        self._index.bulkAdd(index_data)
