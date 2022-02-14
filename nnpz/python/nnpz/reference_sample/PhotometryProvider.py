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

"""
Created on: 29/01/2018
Author: Nikolaos Apostolakos
"""

import os
from typing import Dict, List, Tuple

import astropy.io.fits as fits
import numpy as np
from astropy.io.fits import HDUList
from astropy.table import Table, join
from nnpz.exceptions import MissingDataException, UnknownNameException, \
    WrongFormatException


class PhotometryProvider:
    """
    This is a utility cass for handling NNPZ photometry FITS files
    """

    @staticmethod
    def __check_file_format(filename: str) -> HDUList:
        """
        Checks that the file exists and that it has the correct HDUs
        """

        if not os.path.exists(filename):
            raise FileNotFoundError('File {} does not exist'.format(filename))

        try:
            hdus = fits.open(filename)
        except OSError:
            raise WrongFormatException('Failed to open {} as a FITS file'.format(filename))

        # Check that the first table is the NNPZ photometry
        if 'NNPZ_PHOTOMETRY' not in hdus:
            raise WrongFormatException('File {} does not contain NNPZ photometry'.format(filename))
        return hdus

    @staticmethod
    def __read_filter_transmissions(hdus: HDUList, filter_list: List[str]) -> Dict[str, np.ndarray]:
        """
        Reads the filter transmissions from the hdus
        """

        filter_data = {}
        for name in filter_list:
            if not name in hdus:
                filter_data[name] = None
            else:
                t = Table(hdus[name].data)
                trans = np.ndarray((len(t), 2), dtype=np.float32)
                trans[:, 0] = t.columns[0]
                trans[:, 1] = t.columns[1]
                filter_data[name] = trans
        return filter_data

    @staticmethod
    def __read_photometry_data(phot_table: Table, filter_list: List[str]) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reads from the given table the photometry values for the given filters
        in a numpy array.
        """

        data = np.zeros((len(phot_table), len(filter_list), 2), dtype=np.float32)
        ebv_corr = np.zeros((len(phot_table), len(filter_list)), dtype=np.float32)
        shift_corr = np.zeros((len(phot_table), len(filter_list), 2), dtype=np.float32)
        for i, name in enumerate(filter_list):
            data[:, i, 0] = phot_table[name]
            if name + '_ERR' in phot_table.colnames:
                data[:, i, 1] = phot_table[name + '_ERR']
            if name + '_EBV_CORR' in phot_table.colnames:
                ebv_corr[:, i] = phot_table[name + '_EBV_CORR'].reshape(-1)
            if name + '_SHIFT_CORR' in phot_table.colnames:
                shift_corr[:, i, :] = phot_table[name + '_SHIFT_CORR']
        return data, ebv_corr, shift_corr

    def __init__(self, filename: str, ref_ids: np.ndarray = None):
        """
        Creates a new instance for accessing the given photometry file.

        Args:
            filename: The photometry file to read
            ref_ids: (Optional) the order of the IDs that should be used

        Raises:
            FileNotFoundException: If the file does not exist
            WrongFormatException: If the file is not a NNPZ photometry file
        """

        # Check the format of the file
        hdus = self.__check_file_format(filename)

        # Get the type of photometry in the file
        # pylint: disable=no-member
        self.__type = hdus['NNPZ_PHOTOMETRY'].header.get('PHOTYPE')

        # Create a list with the filters in the file
        # pylint: disable=no-member
        phot_table = Table(hdus['NNPZ_PHOTOMETRY'].data)
        self.__filter_list = [c for c in phot_table.colnames if
                              c != 'ID' and not c.endswith('_ERR') and not c.endswith('_CORR')]

        # Read the filter transmissions from the extension HDUs
        self.__filter_data = self.__read_filter_transmissions(hdus, self.__filter_list)

        # Sort the rows following the ref sample layout
        if ref_ids is not None:
            phot_table = join(dict(ID=ref_ids, position=np.arange(len(ref_ids))), phot_table, 'ID')
            phot_table.sort('position')
            if np.any(phot_table['ID'] != ref_ids):
                raise ValueError('ERROR: Reference sample photometry ID mismatch')

        # Get the IDs
        self.__ids = phot_table['ID']

        # Read the photometry values
        self.__phot_data, self.__ebv_corr_data, self.__shift_corr_data = self.__read_photometry_data(
            phot_table,
            self.__filter_list)
        self.__phot_data.flags.writeable = False
        self.__ebv_corr_data.flags.writeable = False
        self.__shift_corr_data.flags.writeable = False

    def get_type(self) -> str:
        """
        Returns the type of photometry in the file.

        Warnings:
            Since NNPZ 1.0, only F_nu_uJy is supported. Other units could be converted on
            the fly using astropy.units
        """
        return self.__type

    def get_filter_list(self) -> List[str]:
        """
        Returns a list with the available filter names
        """
        return self.__filter_list

    def get_filter_transmission(self, filter_name: str) -> np.ndarray:
        """
        Returns the transmission of the given filter.

        Args:
            filter_name: The name of the filter to get the transmission for

        Returns: A 2D numpy array of single precision floats, where the first
            dimension has size same as the number of knots and the second has
            always size two, where the first element is the wavelength value of
            the knot (expressed in Angstrom) and the second is the filter
            transmission, in the range [0,1].

        Raises:
            UnknownNameException: If the file does not contain photometry for
                the given filter
            MissingDataException: If the HDU for the given filter transmission
                is missing
        """
        if filter_name not in self.__filter_data:
            raise UnknownNameException('Unknown filter {}'.format(filter_name))
        if self.__filter_data[filter_name] is None:
            raise MissingDataException(
                'File does not contain tranmission for {} filter'.format(filter_name))
        return self.__filter_data[filter_name]

    def get_ids(self):
        """
        Returns the IDs of the objects there is photometry in the file
        """
        return self.__ids

    def get_data(self, filter_list: List[str] = None) -> np.ndarray:
        """
        Returns an array with the photometry data for the given bands.

        Args:
            filter_list: The filter names to get the data for. If no filter is
            given, the result is returned for the full filter list, in the same
            order as returned by the getFilterList() method.

        Returns:
            A three-dimensional numpy array of single precision floats where the
            first dimension has the same size as the number of objects the file
            contains photometries for, the second axis has same size as the
            given bands and the third axis has always size two, where the first
            element represents the photometry value and te second the uncertainty.

        Raises:
            UnknownNameException: If the file does not contain photometry for
                any of the given filters

        If the file does not contain uncertainty columns for  some filters, the
        returned array will contain zero values for these uncertainties. The
        order of the second axis of the result is the same as the passed filter
        names.
        """

        # Affected index
        try:
            if filter_list is None or len(filter_list) == 0:
                # numpy will avoid a copy if the index is a slice, while it will *always*
                # copy if it is a list
                filter_idx = slice(len(self.__filter_list))
            else:
                filter_idx = np.array(list(map(self.__filter_list.index, filter_list)))
        except ValueError as e:
            missing = str(e).split()[0]
            raise UnknownNameException('File does not contain photometry for {}'.format(missing))

        # Return a view rather than a copy
        return self.__phot_data[:, filter_idx, :]

    def get_ebv_correction_factors(self, filter_list: List[str] = None) -> np.ndarray:
        """
        Get the EBV correction factors
        """
        try:
            if filter_list is None or len(filter_list) == 0:
                # numpy will avoid a copy if the index is a slice, while it will *always*
                # copy if it is a list
                filter_idx = slice(len(self.__filter_list))
            else:
                filter_idx = np.array(list(map(self.__filter_list.index, filter_list)))
        except ValueError as e:
            missing = str(e).split()[0]
            raise UnknownNameException('File does not contain corrections for {}'.format(missing))

            # Return a view rather than a copy
        return self.__ebv_corr_data[:, filter_idx]

    def get_shift_correction_factors(self, filter_list: List[str] = None) -> np.ndarray:
        """
        Get the filter variation correction factors
        """
        try:
            if filter_list is None or len(filter_list) == 0:
                # numpy will avoid a copy if the index is a slice, while it will *always*
                # copy if it is a list
                filter_idx = slice(len(self.__filter_list))
            else:
                filter_idx = np.array(list(map(self.__filter_list.index, filter_list)))
        except ValueError as e:
            missing = str(e).split()[0]
            raise UnknownNameException('File does not contain corrections for {}'.format(missing))

        # Return a view rather than a copy
        return self.__shift_corr_data[:, filter_idx, :]
