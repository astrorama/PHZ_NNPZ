import os
from typing import Iterable

import numpy as np
from nnpz.exceptions import AlreadySetException
from nnpz.reference_sample import MontecarloDataProvider
from nnpz.reference_sample.BaseProvider import BaseProvider
from nnpz.reference_sample.util import validate_data_files


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

    def initializeFromData(self, object_ids: Iterable[int] = None, data: np.ndarray = None):
        """
        Add new data to the MC provider

        Args:
            object_ids:
                Object ids
            data:
                New data

        Raises:
            AlreadySetException : If the provider has been already initialized
        """
        if len(self._data_map):
            raise AlreadySetException('Provider already initialized')

        if len(data.shape) != 2:
            raise ValueError('Expecting an array with two axes')
        if len(object_ids) != data.shape[0]:
            raise ValueError('The number of objects and shape of the array do not match')

        # Cut data in pieces if necessary
        record_size = data[0].nbytes
        records_per_file = self._data_limit // record_size
        n_files = int(np.ceil(len(object_ids) / records_per_file))

        file_field = f'{self._key}_file'
        offset_field = f'{self._key}_offset'

        index_data = np.zeros(
            (len(object_ids),),
            dtype=[('id', np.int64), (file_field, np.int64), (offset_field, np.int64)]
        )

        for file_i in range(1, n_files + 1):
            data_file = self._data_pattern.format(file_i)
            if os.path.exists(data_file):
                os.unlink(data_file)
            idx = slice((file_i - 1) * records_per_file, file_i * records_per_file)
            index_data['id'] = object_ids[idx]
            index_data[file_field] = file_i
            data_prov = MontecarloDataProvider(data_file)
            index_data[offset_field] = data_prov.append(data[idx])
            data_prov.flush()
            self._data_map[file_i] = data_prov
            self._index.bulkAdd(index_data)
