import os
from typing import Union

import numpy as np
from nnpz.exceptions import AlreadySetException
from nnpz.reference_sample import SedDataProvider, IndexProvider
from nnpz.reference_sample.BaseProvider import BaseProvider
from nnpz.reference_sample.util import validate_data_files, create_new_provider, \
    locate_existing_data_files


class SedProvider(BaseProvider):
    """
    Provides a SED per reference object. SEDs may have different number of knots.
    There is no limitation on the variety of number of knots, but it is expected
    to be limited to a reduced set (i.e. all SEDs of a family normally have the same resolution)

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
        self._prov_for_size = {}
        for data_file in data_files:
            sed_data_prov = SedDataProvider(self._data_pattern.format(data_file))
            self._data_map[data_file] = sed_data_prov
            self._prov_for_size[sed_data_prov.getKnots()] = data_file

    def flush(self):
        """
        Write the changes to disk.
        """
        self._index.flush()
        for data_prov in self._data_map.values():
            data_prov.flush()

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
        if sed_loc:
            return self._data_map[sed_loc.file].readSed(sed_loc.offset)
        return None

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
        loc = self._index.get(obj_id, self._key)
        if loc is not None:
            raise AlreadySetException('SED for ID ' + str(obj_id) + ' is already set')

        current_prov = self._getCurrentDataProvider(data.shape[0])
        new_pos = self._data_map[current_prov].appendSed(data)
        self._index.add(obj_id, self._key, IndexProvider.ObjectLocation(current_prov, new_pos))

    def _getCurrentDataProvider(self, knots):
        """
        Get the index of the active SED provider, create a new one if needed
        """
        if knots not in self._prov_for_size \
            or self._data_map[self._prov_for_size[knots]].size() >= self._data_limit:
            self._prov_for_size[knots] = create_new_provider(
                self._data_map, self._data_pattern, SedDataProvider
            )
        return self._prov_for_size[knots]

    def importData(self, other_index: IndexProvider, data_pattern: str, extra_data: dict):
        """
        Import a set of SED files
        """
        self.extra.update(extra_data)
        other_files = sorted(locate_existing_data_files(data_pattern))
        file_field = f'{self._key}_file'
        offset_field = f'{self._key}_offset'
        for other_sed_i in other_files:
            other_sed_file = data_pattern.format(other_sed_i)
            other_sed_size = os.path.getsize(other_sed_file)
            other_sed = np.load(other_sed_file, mmap_mode='r')

            # Take the part of the index that points to the file other_sed_i
            other_sed_idx_pos = np.where(other_index.raw[file_field] == other_sed_i)[0]
            updated_index = np.array(
                other_index.raw[['id', file_field, offset_field]][other_sed_idx_pos], copy=True)

            # The order of the index does not have to match the order on the file!
            disk_order = np.argsort(updated_index[offset_field])

            # Take the current active provider to store the imported data
            sed_provider_idx = self._getCurrentDataProvider(other_sed.shape[1])
            sed_provider = self._data_map[sed_provider_idx]

            # Fit whatever we can on the current file (approximately)
            available_size = self._data_limit - sed_provider.size()
            nfit = len(other_sed) * min(np.ceil(available_size / other_sed_size), 1)
            updated_index[file_field][:nfit] = sed_provider_idx
            updated_index[offset_field][disk_order[:nfit]] = sed_provider.appendSed(other_sed[:nfit])

            # Create a new one and put in the rest
            sed_provider_idx = self._getCurrentDataProvider(other_sed.shape[1])
            sed_provider = self._data_map[sed_provider_idx]
            updated_index[file_field][nfit:] = sed_provider_idx
            updated_index[offset_field][disk_order[nfit:]] = sed_provider.appendSed(other_sed[nfit:])

            # Update the index
            self._index.bulkAdd(updated_index)
