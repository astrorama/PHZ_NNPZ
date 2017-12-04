"""
Created on: 04/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import os

import numpy as np

from nnpz.photometry import FilterProviderInterface
from nnpz.exceptions import *


class DirectoryFilterProvider(FilterProviderInterface):
    """Implementation of the FilterProviderInterface reading files from a directory.

    The directory can contain filter transmissions as ASCII tables of
    two columns, the first of which represents the wavelength (expressed in
    Angstrom) and the second one the filter transmission (in the range [0,1]).
    The names of the filters are same as the file names, without the extension.

    The directory can (optionally) contain the file filter_list.txt, which can
    be used for defining the order of the filters and for changing their names.
    Each line of this file follows the format "FILENAME [: FILTER_NAME]". The
    filter name is optional and if missing the filename is used instead (without
    the extension).
    """


    def __parseFilterListFile(self, path, dir_contents):
        """Parses the filter_list.txt in a list of (filtername, filename) pairs"""

        with open(os.path.join(path, 'filter_list.txt')) as f:
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
                filename = l
                filtername = os.path.splitext(l)[0]

            if not os.path.exists(os.path.join(path, filename)):
                raise FileNotFoundException('Missing filter transmission: ' + filename)
            result.append((filtername, filename))

        return result


    def __init__(self, path):
        """Creates a new DirectoryFilterProvider for the given directory.

        Args:
            path: The directory to read the filters from

        Raises:
            InvalidPathException: If there is no such directory
            FileNotFoundException: If a file in the filter_list.txt does not exist
        """
        if not os.path.isdir(path):
            raise InvalidPathException(path + ' is not a directory')

        # Get the list with the (filtername, filename) pairs
        dir_contents = os.listdir(path)
        filter_file_pairs = (self.__parseFilterListFile(path, dir_contents)
                 if ('filter_list.txt' in dir_contents)
                 else [(os.path.splitext(f)[0], f) for f in dir_contents])

        # Populate the member variables
        self.__name_list = []
        self.__file_map = {}
        self.__data_map = {}
        for filtername, filename in filter_file_pairs:
            self.__name_list.append(filtername)
            self.__file_map[filtername] = os.path.join(path, filename)


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