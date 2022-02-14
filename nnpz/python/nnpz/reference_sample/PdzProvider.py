#
# Copyright (C) 2012-2022 Euclid Science Ground Segment
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

from typing import List, Union

import numpy as np
from nnpz.exceptions import InvalidAxisException, InvalidDimensionsException
from nnpz.reference_sample import IndexProvider, PdzDataProvider
from nnpz.reference_sample.BaseProvider import BaseProvider
from nnpz.reference_sample.util import locate_existing_data_files, validate_data_files


class PdzProvider(BaseProvider):
    """
    Provides a PDZ per reference object. The PDZ has a fixed number of bins for all objects.

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

    def _swap_provider(self, index):
        if index != self._current_data_index:
            if self._current_data_provider is not None:
                self._current_data_provider.flush()
            self._current_data_index = index
            self._current_data_provider = PdzDataProvider(
                self._data_pattern.format(index)
            )

    def get_redshift_bins(self) -> np.ndarray:
        if self._current_data_provider is None:
            self._swap_provider(1)
        return self._current_data_provider.get_redshift_bins()

    def get_pdz_for_index(self, obj_idxs: np.ndarray) -> np.ndarray:
        # Access following the physical layout
        sorted_order = np.argsort(obj_idxs)
        ids = self._index.get_ids()[obj_idxs[sorted_order]]
        output = np.zeros((len(obj_idxs), len(self.get_redshift_bins())), dtype=np.float32)
        for i, obj_id in zip(sorted_order, ids):
            output[i] = self.get_pdz_data(obj_id)
        return output

    def get_pdz_data(self, obj_id: int) -> Union[np.ndarray, None]:
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
        pdz_loc = self._index.get(obj_id, self._key)
        if not pdz_loc:
            return None
        if not self._current_data_index or self._current_data_index != pdz_loc.file:
            self._swap_provider(pdz_loc.file)
        return self._current_data_provider.read_pdz(pdz_loc.offset)

    def _get_writeable_pdz_provider(self, binning):
        """
        Get the index of the last writeable PDZ provider, create a new one if needed
        """
        if not self._last_data_index:
            self._last_data_index = 1
        self._swap_provider(self._last_data_index)

        # Check if the last file exceeded the size limit and create a new one
        if self._current_data_provider.size() >= self._data_limit:
            self._last_data_index += 1
            self._swap_provider(self._last_data_index)

        # Set, or crosscheck, the binning
        existing_zs = self._current_data_provider.get_redshift_bins()
        if existing_zs is None:
            self._current_data_provider.set_redshift_bins(binning)
        elif not np.array_equal(binning, existing_zs):
            raise InvalidAxisException('Given wavelengths are different than existing ones')

        return self._last_data_index, self._current_data_provider

    def import_data(self, other_index: IndexProvider, data_pattern: str, extra_data: dict):
        """
        Import a set of PDZ files
        """
        self.extra.update(extra_data)
        other_files = sorted(locate_existing_data_files(data_pattern))
        for other_pdz_i in other_files:
            other_pdz_file = data_pattern.format(other_pdz_i)
            other_pdz = np.load(other_pdz_file, mmap_mode='r')

            # Ask for the IDs following disk order
            other_ids = other_index.get_ids_for_file(other_pdz_i, self._key)

            # Import the data
            self.add_data(other_ids, other_pdz)

    def add_data(self, object_ids: List[int] = None, data: np.ndarray = None):
        """
        Add new data to the PDZ provider

        Args:
            object_ids:
                Object ids
            data:
                New data. The first entry must correspond to the PDZ bins
        """
        if len(data.shape) != 2:
            raise InvalidDimensionsException('The PDZ data must have two axes')
        if len(object_ids) != data.shape[0] - 1:
            raise InvalidDimensionsException(
                'The number of PDZ entries does not match the number of objects'
            )

        pdz_bins = data[0]
        pdz = data[1:]

        record_size = pdz[0].nbytes
        records_per_file = self._data_limit // record_size + (self._data_limit % record_size > 0)

        # Index
        file_field = f'{self._key}_file'
        offset_field = f'{self._key}_offset'

        index_data = np.zeros(
            (len(object_ids),),
            dtype=[('id', np.int64), (file_field, np.int64), (offset_field, np.int64)]
        )

        # First available provider and merge whatever is possible
        provider_idx, provider = self._get_writeable_pdz_provider(pdz_bins)
        available_size = self._data_limit - provider.size()
        nfit = available_size // record_size + (available_size % record_size > 0)

        index_data['id'][:nfit] = object_ids[:nfit]
        index_data[file_field][:nfit] = provider_idx
        index_data[offset_field][:nfit] = provider.append_pdz(pdz[:nfit])
        provider.flush()

        # Cut what's left in whole files
        n_files = int(np.ceil(len(object_ids[nfit:]) / records_per_file))

        for file_i in range(n_files):
            selection = slice(nfit + records_per_file * file_i,
                              nfit + records_per_file * (file_i + 1))

            provider_idx, provider = self._get_writeable_pdz_provider(pdz_bins)
            index_data['id'][selection] = object_ids[selection]
            index_data[file_field][selection] = provider_idx
            index_data[offset_field][selection] = provider.append_pdz(pdz[selection])
            provider.flush()

        self._index.bulk_add(index_data)
