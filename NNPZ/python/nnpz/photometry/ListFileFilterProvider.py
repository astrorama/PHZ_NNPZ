"""
Created on: 18/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import os

import numpy as np

from nnpz.photometry import FilterProviderInterface
from nnpz.exceptions import *


class ListFileFilterProvider(FilterProviderInterface):
    """Implementation of the FilterProviderInterface getting the list of filters from a file.

    The list file can contain the filter transmission file names as absolute
    paths or as relative paths to the list file. Empty lines are ignored and
    comments are indicated with the '#' character. Each line follows the format
    "FILENAME [: FILTER_NAME]". The filter name is optional and if missing the
    filename is used instead (without the extension).

    The filter transmission files are ASCII tables of two columns, the first of
    which represents the wavelength (expressed in Angstrom) and the second one
    the filter transmission (in the range [0,1]).
    """


    def __parseFilterListFile(self, list_file):
        """Parses the given file in a list of (filtername, filename) pairs"""

        list_file = os.path.abspath(list_file)

        with open(list_file) as f:
            lines = f.readlines()

        result = []

        for l in lines:
            # Remove any comments from the line
            if '#' in l:
                l = l[:l.index('#')]
            l = l.strip()
            if len(l) == 0:
                continue

            # Check if the user gave a name
            if ':' in l:
                filename, filtername = l.split(':')
                filename = filename.strip()
                filtername = filtername.strip()
            else:
                filename = l.strip()
                filtername = os.path.splitext(os.path.basename(filename))[0]

            # If we have a relative path, make it relative to the list file
            if not os.path.isabs(filename):
                filename = os.path.join(os.path.dirname(list_file), filename)

            if not os.path.exists(filename):
                raise FileNotFoundException('Missing filter transmission: ' + filename)
            result.append((filtername, filename))

        return result


    def __init__(self, list_file):
        """Creates a new ListFileFilterProvider for the given list file.

        Args:
            list_file: The file containing the list of the filters

        Raises:
            FileNotFoundException: If the list_file does not exist
            FileNotFoundException: If a file in the list_file does not exist
        """
        if not os.path.exists(list_file):
            raise FileNotFoundException(list_file + ' does not exist')

         # Get the list with the (filtername, filename) pairs
        filter_file_pairs = self.__parseFilterListFile(list_file)

        # Populate the member variables
        self.__name_list = []
        self.__file_map = {}
        self.__data_map = {}
        for filtername, filename in filter_file_pairs:
            self.__name_list.append(filtername)
            self.__file_map[filtername] = filename


    def getFilterNames(self):
        """Returns the names of the filters"""
        return self.__name_list


    def getFilterTransmission(self, name):
        """Provides the transmission curve of the filter with the given name.

        Args:
            name: The name of the filter to get the transmission for

        Returns:
            A 2D numpy array of single precision floating point numbers. The
            first dimension represents the knots of the filter transmission
            and the second one has always size 2, representing the wavelength
            (expressed in Angstrom) and the transmission value (in the range
            [0,1]).

        Raises:
            UnknownNameException: If there is no filter with the given name
        """

        # First check if we have already read the data from the file
        if name in self.__data_map:
            return self.__data_map[name]

        # Here we need to read the data from the file
        if not name in self.__file_map:
            raise UnknownNameException('Unknown filter:' + name)

        from astropy.table import Table
        table = Table.read(self.__file_map[name], format='ascii')
        data = np.ndarray((len(table), 2), dtype='float32')
        data[:,0] = table.columns[0]
        data[:,1] = table.columns[1]

        self.__data_map[name] = data
        return data