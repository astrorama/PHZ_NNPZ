from typing import List, Union

import numpy as np
from nnpz.exceptions import InvalidDimensionsException
from nnpz.reference_sample import IndexProvider, SedDataProvider
from nnpz.reference_sample.BaseProvider import BaseProvider
from nnpz.reference_sample.util import locate_existing_data_files, validate_data_files


class SedProvider(BaseProvider):
    """
    Provides a SED per reference object. SEDs may have different number of knots.
    There is no limitation on the variety of number of knots, but it is expected
    to be limited to a reduced set (i.e. all SEDs of a family normally have the same resolution)

    The SedProvider will keep at least one file per number of knots open

    Args:
        *args:
            Forwarded to BaseProvider
        **kwargs:
            Forwarded to BaseProvider
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_files = validate_data_files(self._data_pattern, self._index, self._key)
        self._current_index_for_knots = dict()
        self._data_map = dict()
        self._last_index_for_knots = dict()
        for data_file in self._data_files:
            sed_data_prov = SedDataProvider(self._data_pattern.format(data_file))
            self._last_index_for_knots[sed_data_prov.getKnots()] = data_file
        self._last_data_index = max(self._data_files) if self._data_files else 0

    def flush(self):
        """
        Write the changes to disk.
        """
        self._index.flush()
        for provider in self._data_map.values():
            provider.flush()

    def _swapProvider(self, index: int):
        if index in self._data_map:
            return

        # Load provider
        data_provider = SedDataProvider(self._data_pattern.format(index))
        knots = data_provider.getKnots()

        # Close previous for the same size
        prev_index = self._current_index_for_knots.get(knots, None)
        if prev_index is not None:
            del self._data_map[prev_index]

        # Store reference to the new one
        self._current_index_for_knots[knots] = index
        self._data_map[index] = data_provider

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
        sed_loc = self._index.get(obj_id, self._key)
        if not sed_loc:
            return None
        self._swapProvider(sed_loc.file)
        return self._data_map[sed_loc.file].readSed(sed_loc.offset)

    def _getWriteableDataProvider(self, knots):
        """
        Get the index of the active SED provider, create a new one if needed
        """
        if knots not in self._last_index_for_knots:
            self._last_data_index += 1
            self._last_index_for_knots[knots] = self._last_data_index
        self._swapProvider(self._last_index_for_knots[knots])

        # Check if the last file exceeded the size limit and create a new one
        index = self._last_index_for_knots[knots]
        if self._data_map[index].size() >= self._data_limit:
            self._last_data_index += 1
            self._last_index_for_knots[knots] = self._last_data_index
            self._swapProvider(self._last_data_index)
            index = self._last_index_for_knots[knots]

        return self._last_index_for_knots[knots], self._data_map[index]

    def importData(self, other_index: IndexProvider, data_pattern: str, extra_data: dict):
        """
        Import a set of SED files
        """
        self.extra.update(extra_data)
        other_files = sorted(locate_existing_data_files(data_pattern))

        for other_sed_i in other_files:
            other_sed_file = data_pattern.format(other_sed_i)
            other_sed = np.load(other_sed_file, mmap_mode='r')

            # Ask for the IDs following disk order
            other_ids = other_index.getIdsForFile(other_sed_i, self._key)

            # Import the data
            self.addData(other_ids, other_sed)

    def addData(self, object_ids: List[int] = None, data: np.ndarray = None):
        """
        Add new data to the SED provider

        Args:
            object_ids:
                Object IDs
            data:
                SED data
        """
        if len(data.shape) != 3:
            raise InvalidDimensionsException('The SED data must have three axes')
        if len(object_ids) != data.shape[0]:
            raise InvalidDimensionsException(
                'The number of SED entries does not match the number of objects'
            )

        record_size = data[0].nbytes

        # Index
        file_field = f'{self._key}_file'
        offset_field = f'{self._key}_offset'

        index_data = np.zeros(
            (len(object_ids),),
            dtype=[('id', np.int64), (file_field, np.int64), (offset_field, np.int64)]
        )

        # Take the current active provider to store the imported data
        sed_provider_idx, sed_provider = self._getWriteableDataProvider(data.shape[1])

        # Fit whatever we can on the current file (approximately)
        available_size = self._data_limit - sed_provider.size()
        nfit = int(len(data) * min(np.ceil(available_size / record_size), 1))
        index_data['id'][:nfit] = object_ids[:nfit]
        index_data[file_field][:nfit] = sed_provider_idx
        index_data[offset_field][:nfit] = sed_provider.appendSed(data[:nfit])
        sed_provider.flush()

        # Create a new one and put in the rest
        sed_provider_idx, sed_provider = self._getWriteableDataProvider(data.shape[1])
        index_data['id'][nfit:] = object_ids[nfit:]
        index_data[file_field][nfit:] = sed_provider_idx
        index_data[offset_field][nfit:] = sed_provider.appendSed(data[nfit:])
        sed_provider.flush()

        # Update the index
        self._index.bulkAdd(index_data)
