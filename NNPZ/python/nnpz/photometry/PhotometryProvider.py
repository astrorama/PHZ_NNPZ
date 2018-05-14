"""
Created on: 29/01/2018
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import os
import numpy as np
import astropy.io.fits as fits
from astropy.table import Table

from nnpz.exceptions import *


class PhotometryProvider(object):
    """This is a utility cass for handling NNPZ photometry FITS files"""


    def __checkFileFormat(self, filename):
        """Checks that the file exists and that it has the correct HDUs"""

        if not os.path.exists(filename):
            raise FileNotFoundException('File {} does not exist'.format(filename))

        try:
            hdus = fits.open(filename)
        except:
            raise WrongFormatException('Failed to open {} as a FITS file'.format(filename))

        # Check that the first table is the NNPZ photometry
        if not 'NNPZ_PHOTOMETRY' in hdus:
            raise WrongFormatException('File {} does not contain NNPZ photometry'.format(filename))
        return hdus


    def __readFilterTransmissions(self, hdus, filter_list):
        """Reads the filter transmissions from the hdus"""

        filter_data = {}
        for name in filter_list:
            if not name in hdus:
                filter_data[name] = None
            else:
                t = Table(hdus[name].data)
                trans = np.ndarray((len(t),2), dtype=np.float32)
                trans[:,0] = t.columns[0]
                trans[:,1] = t.columns[1]
                filter_data[name] = trans
        return filter_data


    def __readPhotometryData(self, phot_table, filter_list):
        """Reads from the given table the photometry values for the given filters
        in a numpy array."""

        data = np.zeros((len(phot_table), len(filter_list), 2), dtype=np.float32)
        for i, name in enumerate(filter_list):
            data[:,i,0] = phot_table[name]
            if name+'_ERR' in phot_table.colnames:
                data[:,i,1] = phot_table[name+'_ERR']
        return data


    def __init__(self, filename):
        """Creates a new instance for accessing the given photometry file.

        Args:
            filename: The photometry file to read

        Raises:
            FileNotFoundException: If the file does not exist
            WrongFormatException: If the file is not a NNPZ photometry file
        """

        # Check the format of the file
        hdus = self.__checkFileFormat(filename)

        # Get the type of photometry in the file
        self.__type = hdus[1].header.get('PHOTYPE')

        # Create a list with the filters in the file
        phot_table = Table(hdus['NNPZ_PHOTOMETRY'].data)
        self.__filter_list = [c for c in phot_table.colnames if c != 'ID' and not c.endswith('_ERR')]

        # Read the filter transmissions from the extension HDUs
        self.__filter_data = self.__readFilterTransmissions(hdus, self.__filter_list)

        # Get the IDs
        self.__ids = phot_table['ID']

        # Read the photometry values
        self.__phot_data = self.__readPhotometryData(phot_table, self.__filter_list)


    def getType(self):
        """Returns the type of photometry in the file.

        The possible types are:
        - Photons: The photometry values are photon count rates, expressed in counts/s/cm^2
        - F_nu: The photometry values are energy flux densities expressed in erg/s/cm^2/Hz
        -F_nu_uJy: The photometry values are energy flux densities expressed in uJy
        -F_lambda: The photometry values are energy fluxes densities expressed in erg/s/cm^2/A
        -MAG_AB: The photometry values are AB magnitudes
        """
        return self.__type


    def getFilterList(self):
        """Returns a list with the available filter names"""
        return self.__filter_list


    def getFilterTransmission(self, filter):
        """Returns the transmission of the given filter.

        Args:
            filter: The name of the filter to get the transmission for

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
        if not filter in self.__filter_data:
            raise UnknownNameException('Unknown filter {}'.format(filter))
        if self.__filter_data[filter] is None:
            raise MissingDataException('File does not contain tranmission for {} filter'.format(filter))
        return self.__filter_data[filter]


    def getIds(self):
        """Returns the IDs of the objects there is photometry in the file"""
        return self.__ids


    def getData(self, *filter_list):
        """Returns an array with the photometry data for the given bands.

        Args:
            filter_list: The filter names to get the data for. If no filter is
            given, the result is returned for the full filter list, in the same
            order as returned by the getFilterList() method.

        Returns:
            A three dimensional numpy array of single precision floats where the
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

        if len(filter_list) == 0:
            filter_list = self.getFilterList()

        # Create the array to store the results
        result = np.zeros((len(self.__ids), len(filter_list), 2), dtype=np.float32)

        for user_i, name in enumerate(filter_list):
            # Find the index of the filter
            try:
                local_i = self.__filter_list.index(name)
            except ValueError:
                raise UnknownNameException('File does not contain photometry for {}', name)

            # Populate the result
            result[:,user_i,:] = self.__phot_data[:,local_i,:]

        return result