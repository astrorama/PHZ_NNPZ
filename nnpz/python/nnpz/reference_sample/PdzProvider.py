import os
import pathlib
from typing import Union

import numpy as np
from nnpz.exceptions import InvalidAxisException, AlreadySetException, InvalidDimensionsException
from nnpz.reference_sample import PdzDataProvider, IndexProvider
from nnpz.reference_sample.BaseProvider import BaseProvider
from nnpz.reference_sample.util import validate_data_files, create_new_provider, \
    locate_existing_data_files


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
        data_files = validate_data_files(self._data_pattern, self._index, 'PdzProvider')

        self._data_map = {}
        for data_file in data_files:
            self._data_map[data_file] = PdzDataProvider(self._data_pattern.format(data_file))

    def flush(self):
        """
        Write the changes to disk.
        """
        self._index.flush()
        for data_prov in self._data_map.values():
            data_prov.flush()

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
        pdz_loc = self._index.get(obj_id)
        if not pdz_loc:
            return None
        z_bins = self._data_map[pdz_loc.file].getRedshiftBins().reshape(-1, 1)
        pdz_data = self._data_map[pdz_loc.file].readPdz(pdz_loc.offset).reshape(-1, 1)
        return np.hstack([z_bins, pdz_data])

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
        loc = self._index.get(obj_id)
        if loc is not None:
            raise AlreadySetException('PDZ for ID ' + str(obj_id) + ' is already set')

        # Convert the data to a numpy array for easier handling
        data_arr = np.asarray(data, dtype=np.float32)
        if len(data_arr.shape) != 2 or data_arr.shape[1] != 2:
            raise InvalidDimensionsException()

        pdz_file = self._getCurrentPdzProvider(data_arr[:, 0])

        # Add the PDZ data in the last file, normalizing first
        integral = np.trapz(data_arr[:, 1], data_arr[:, 0])
        new_pos = self._data_map[pdz_file].appendPdz(data_arr[:, 1] / integral)
        self._index.add(obj_id, IndexProvider.ObjectLocation(pdz_file, new_pos))

    def _getCurrentPdzProvider(self, binning):
        """
        Get the index of the active PDZ provider, create a new one if needed
        """
        if self._data_map:
            last_pdz_file = max(self._data_map)
        else:
            last_pdz_file = create_new_provider(
                self._data_map, self._data_pattern, PdzDataProvider
            )

        # Check if the last file exceeded the size limit and create a new one
        if self._data_map[last_pdz_file].size() >= self._data_limit:
            last_pdz_file = create_new_provider(
                self._data_map, self._data_pattern, PdzDataProvider
            )

        # Set, or crosscheck, the binning
        existing_zs = self._data_map[last_pdz_file].getRedshiftBins()
        if existing_zs is None:
            self._data_map[last_pdz_file].setRedshiftBins(binning)
        elif not np.array_equal(binning, existing_zs):
            raise InvalidAxisException('Given wavelengths are different than existing ones')

        return last_pdz_file

    def importData(self, index_path: Union[str, pathlib.Path], data_pattern: str, extra_data: dict):
        """
        Import a set of PDZ files
        """
        self.extra.update(extra_data)
        other_index = np.load(index_path, mmap_mode='r')
        other_files = sorted(locate_existing_data_files(data_pattern))
        for other_pdz_i in other_files:
            other_pdz_file = data_pattern.format(other_pdz_i)
            other_pdz_size = os.path.getsize(other_pdz_file)
            other_pdz = np.load(other_pdz_file, mmap_mode='r')

            # Take the part of the index that points to the file other_pdz_i
            other_pdz_index_pos = np.nonzero(other_index[:, 1] == other_pdz_i)[0]
            updated_index = np.array(other_index[other_pdz_index_pos], copy=True)

            # Take the current active provider to store the imported data
            pdz_provider_idx = self._getCurrentPdzProvider(other_pdz[0, :])
            pdz_provider = self._data_map[pdz_provider_idx]

            # Fit whatever we can on the current file (approximately)
            available_size = self._data_limit - pdz_provider.size()
            nfit = len(other_pdz) * min(np.ceil(available_size / other_pdz_size), 1)
            updated_index[:nfit, 1] = pdz_provider_idx
            updated_index[:nfit, 2] = pdz_provider.appendPdz(other_pdz[1:nfit])

            # Create a new one and put in the rest
            pdz_provider_idx = self._getCurrentPdzProvider(other_pdz[0, :])
            pdz_provider = self._data_map[pdz_provider_idx]
            updated_index[nfit:, 1] = pdz_provider_idx
            updated_index[nfit:, 2] = pdz_provider.appendPdz(other_pdz[nfit:])

            # Update the index
            self._index.bulkAdd(updated_index)
