"""
Created on: 09/11/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import os
from astropy.table import Table
import astropy.io.fits as fits
import numpy as np

from nnpz.utils import Logging


logger = Logging.getLogger('ReferenceSample')

_index_filename = 'index.fits'
_sed_data_filename = 'sed_data.bin'
_pdz_daa_filename = 'pdz_data.bin'


class ReferenceSample(object):
    """Object for handling the reference sample format of NNPZ"""
    
    
    @staticmethod
    def createNew(path):
        """Creates a new reference sample directory.

        Args:
            path: The path to create the reference sample in
            
        Returns:
            An instance of the RefernceSample class representing the new sample

        The newly created reference sample directory will conain an index file
        table with no rows, an empty SED data file and an empty PDZ data file.
        """
        
        logger.debug('Creating reference sample directory ' + path + '...')
        
        # Create the directory
        os.makedirs(path)
        
        # Create an empty index file
        index = Table(dtype=[('ID', 'int64'), ('SED_POS', 'int64'), ('PDZ_POS', 'int64')])
        index.write(os.path.join(path, _index_filename), format='fits')
        
        # Create  empty sed and pdz data files
        open(os.path.join(path, _sed_data_filename), 'wb').close()
        open(os.path.join(path, _pdz_daa_filename), 'wb').close()
    
    
    def reload(self):
        """Reloads the reference sample from the drive"""
        return
    
    
    def getIds(self):
        """Returns the IDs of the reference sample objects as a numpy array of
        double precission (8 bytes) intengers"""
        return
    
    
    def createObject(self, id):
        """Creates a new object in the reference sample.
        
        Args:
            id: The ID of the object to create. It must be an integer.
        
        Throws:
            DuplicateIdException: if there is already an object with this ID in
                the reference sample
        
        When this method is called a new object in the reference sample is
        created.
        """
        return
    
    
    def addSedData(self, id, data):
        """Adds the SED data of a reference sample object.
        
        Args:
            id: The ID of the object to add the SED for. It must be an integer.
            data:
        """
    
    