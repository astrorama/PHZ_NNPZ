"""
Created on: 15/11/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import os
from astropy.table import Table
import astropy.io.fits as fits
import numpy as np

from nnpz.exceptions import *


class PhotometryProvider(object):
    """Class for handling the photometry files in a NNPZ directory.
    
    Photometry files are considered all FITS tables in the reference sampe
    directory. This class organizes the access to the bands either grouped by
    file, or as individual bands.
    """
    
    
    def __isPhotometryFile(self, filename):
        """Checks if the given file is a photometry FITS table"""
        
        try:
            hdu_list = fits.open(filename)
            if hdu_list[1].name == 'NNPZ_PHOTOMETRY':
                return True
        except:
            pass
        return False
    
    
    def __findPhotometryFiles(self, path):
        """Returns a list with the photometry FITS files in the given path"""
        
        filename_list = [os.path.join(path, f) for f in os.listdir(path) if f != 'index.fits'
                                            and f != 'sed_data.bin' and f != 'pdz_data.bin']
        return [p for p in filename_list if self.__isPhotometryFile(p)]
    
    
    def __init__(self, path):
        """Creates a new instance handling the photometry files in the given path.
        
        Args:
            path: The path to look for photometry files in
        
        Raises:
            IdMismatchException: If not all photometry files have the same ID
                columns
        """
        
        # A map where all the data are stored
        self.__data_map = {}
        # A map keeping track in which files a band is stored
        self.__band_map = {}
        # A list with all the files
        self.__file_list = []
        # A map with all the bands in a file
        self.__file_bands_map = {}
        
        self.__ids = None
        self.__file_list = []
        for f_abs in self.__findPhotometryFiles(path):
            f = os.path.basename(f_abs)
            self.__file_list.append(f)
            photo_table = Table.read(f_abs)
            
            # Check if the IDs are the same in all files
            if self.__ids is None:
                self.__ids = photo_table['ID'].data
            elif not np.array_equal(self.__ids, photo_table['ID'].data):
                raise IdMismatchException('IDs in file ' + f + ' are different')
            
            # Get the band names and populate the band_map
            self.__file_bands_map[f] = photo_table.colnames[1:]
            for b in self.__file_bands_map[f]:
                if not b in self.__band_map:
                    self.__band_map[b] = set()
                self.__band_map[b].add(f)
            
            # Populate the data_map
            for b in self.__file_bands_map[f]:
                self.__data_map[(f, b)] = np.asarray(photo_table[b].data, dtype=np.float32)
    
    
    def getFilenameList(self):
        """Returns a list of the available photometry files.
        
        Returns: A list of strings with the FITS table file names
        """
        return self.__file_list
    
    
    def getFileBandList(self, filename):
        """Returns the bands for which photometry is available in a specific file.
        
        Args:
            filename: The name of the file to get the avaliable bands for
        
        Returns: A list of strings with the available band names
        
        Raises:
            UnknownNameException: If there is no such photometry file
        """
        
        if filename in self.__file_bands_map:
            return self.__file_bands_map[filename]
        else:
            raise UnknownNameException()
        
        
    def getFullBandList(self):
        """Returns a list of all bands avaliable.
        
        Returns:
            A list of tuples where the first element is the name of the file the
            band was found in and the second element the name of the band itself
        """
        return self.__data_map.keys()
    
    
    def getBandsData(self, *band_list):
        """Returns the photometry data for the requested bands.
        
        Args:
            args: The bands to return the data for. They can be either single
                string representing the name of the band or a tuple (x,y) where
                x is the name of the file and y the name of the band.
        
        Returns:
            A two dimensional array where the first axis represents the band and
            the second the photometry values. The order of the bands is the one
            of the given arguments.
        
        Raises:
            UnknownNameException: If a given band is not in the photometry files
            AmbiguityException: If a band is given as a single string and it is
                contained in more than one files
        """
        
        result = np.ndarray((len(band_list), len(self.__ids)), dtype=np.float32)
        
        for i, band in enumerate(band_list):
            
            # If te user gave just a band name, fill the file information
            if isinstance(band, str):
                f_list = self.__band_map.get(band)
                if f_list is None:
                    raise UnknownNameException('Unknown band ' + band)
                if len(f_list) > 1:
                    raise AmbiguityException('Band ' + band + ' provided by muliple files ' + str(f_list))
                band = (next(iter(f_list)), band)
            
            # Fill the slice of the result corresponding to the band
            if not band in self.__data_map:
                raise UnknownNameException('Unknown band ' + str(band))
            result[i,:] = self.__data_map[band]
            
        return result
        
        
    def getFileData(self, filename):
        """Returns he photometry data for the bands of a specific file.
        
        Args:
            filename: The name of the photometry file
            
        Returns:
            A two dimensional array where the first axis represents the band and
            the second the photometry values. The order of the bands is the same
            as the result of the getBandListInFile() method.
        
        Raises:
            UnknownNameException: If the given filename is not a photometry file
        """
        bands = [(filename, b) for b in self.getFileBandList(filename)]
        return self.getBandsData(*bands)
    
    
    def validate(self, id_list):
        """Checks if the photometry files are consistent with the given ID list.
        
        Returns: True if the IDs of all the photometry files match the given
            list, False otherwise
        """
        return np.array_equal(id_list, self.__ids)