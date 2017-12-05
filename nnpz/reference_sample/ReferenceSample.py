"""
Created on: 09/11/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import os
import numpy as np

from nnpz.utils import Logging
from nnpz.exceptions import *
from nnpz.reference_sample import IndexProvider, PdzDataProvider, SedDataProvider


logger = Logging.getLogger('ReferenceSample')

_index_filename = 'index.bin'
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

        The newly created reference sample directory will conain an empty index
        file, an empty SED data file and an empty PDZ data file.
        """
        
        logger.debug('Creating reference sample directory ' + path + '...')
        
        # Create the directory
        os.makedirs(path)
        
        # Create  empty index, sed and pdz data files
        open(os.path.join(path, _index_filename), 'wb').close()
        open(os.path.join(path, _sed_data_filename), 'wb').close()
        open(os.path.join(path, _pdz_daa_filename), 'wb').close()
        
        return ReferenceSample(path)
        
        
    def __init__(self, path):
        """Creates a new ReferenceSample instance, managing the given path.
        
        Args:
            path: The path of the reference sample
            
        Raises:
            FileNotFoundException: If any of the index or data files are not
                present in the given directory directory
        """
        
        # Construct the paths to all files
        self.__root_path = path
        self.__index_path = os.path.join(self.__root_path, 'index.bin')
        self.__sed_path = os.path.join(self.__root_path, 'sed_data.bin')
        self.__pdz_path = os.path.join(self.__root_path, 'pdz_data.bin')
        
        # Check that all files exist
        if not os.path.exists(self.__root_path):
            raise FileNotFoundException(self.__root_path + ' does not exist')
        if not os.path.exists(self.__index_path):
            raise FileNotFoundException(self.__index_path + ' does not exist')
        if not os.path.exists(self.__sed_path):
            raise FileNotFoundException(self.__sed_path + ' does not exist')
        if not os.path.exists(self.__pdz_path):
            raise FileNotFoundException(self.__pdz_path + ' does not exist')
        
        # Now call the reload() method, which will setup all the providers to
        # delegate the file handling to
        self.reload()
    
    
    def reload(self):
        """Reloads the reference sample from the drive"""
        
        # Initialize the internal objects that encapsulate the handling of the
        # reference sample files
        self.__index = IndexProvider(self.__index_path)
        self.__sed = SedDataProvider(self.__sed_path)
        self.__pdz = PdzDataProvider(self.__pdz_path)
        
        
    def size(self):
        """Returns the number of objects in the reference sample"""
        return self.__index.size()
    
    
    def getIds(self):
        """Returns the IDs of the reference sample objects as a numpy array of
        double precission (8 bytes) intengers"""
        return self.__index.getIdList()
    
    
    def getSedData(self, obj_id):
        """Returns the SED data for the given reference sample object.
        
        Args:
            obj_id: The ID of the object to retrieve the SED for
        
        Returns:
            None if the SED is not set for the given object, otherwise the data
            of the SED as a two dimensional numpy array of single precission
            floats. The first dimension has size same as the number of the knots
            and the second dimension has always size equal to two, with the
            first element representing the wavelength and the second the energy
            value.
        
        Raises:
            IdMismatchException: If there is no such ID in the reference sample
            CorruptedFileException: If the ID stored in the index file is
                different than the one stored in the SED data file
        """
        
        # Get from the index the position in the SED data file
        sed_pos, _ = self.__index.getPositions(obj_id)
        if sed_pos == -1:
            return None
        
        # Read the data from the SED data file
        file_id, sed_data = self.__sed.readSed(sed_pos)
        
        # Check that the index and the SED data file are consistent
        if file_id != obj_id:
            raise CorruptedFileException('Corrupted files: Index file contains ' +
                        ' the ID ' + str(obj_id) + ' and SED data file the ' + str(file_id))
                        
        return sed_data
    
    
    def getPdzData(self, obj_id):
        """Returns the PDZ data for the given reference sample object.
        
        Args:
            obj_id: The ID of the object to retrieve the PDZ for
        
        Returns:
            None if the PDZ is not set for the given object, otherwise the data
            of the PDZ as a two dimensional numpy array of single precission
            floats. The first dimension has size same as the number of the knots
            and the second dimension has always size equal to two, with the
            first element representing the wavelength and the second the
            probability value.
        
        Raises:
            IdMismatchException: If there is no such ID in the reference sample
            CorruptedFileException: If the ID stored in the index file is
                different than the one stored in the PDZ data file
        """
        
        # Get from the index the position in the PDZ data file
        _, pdz_pos = self.__index.getPositions(obj_id)
        if pdz_pos == -1:
            return None
                        
        # Get the redshift bins of the PDZ. Note that this should never return
        # None if the pdz_pos is not -1.
        z_bins = self.__pdz.getRedshiftBins()
        
        # Read the data from the SED data file
        file_id, pdz_data = self.__pdz.readPdz(pdz_pos)
        
        # Check that the index and the PDZ data file are consistent
        if file_id != obj_id:
            raise CorruptedFileException('Corrupted files: Index file contains ' +
                        ' the ID ' + str(obj_id) + ' and PDZ data file the ' + str(file_id))
                        
        result = np.ndarray((len(z_bins), 2), dtype=np.float32)
        result[:,0] = z_bins
        result[:,1] = pdz_data
        return result
    
    
    def createObject(self, new_id):
        """Creates a new object in the reference sample.
        
        Args:
            new_id: The ID of the object to create. It must be an integer.
        
        Throws:
            DuplicateIdException: if there is already an object with this ID in
                the reference sample
        
        When this method is called a new object in the reference sample is
        created. No SED or PDZ data are assosiated with this object. They can be
        set using the addSedData() and addPdzData() methods.
        """
        self.__index.appendId(new_id)
    
    
    def addSedData(self, obj_id, data):
        """Adds the SED data of a reference sample object.
        
        Args:
            obj_id: The ID of the object to add the SED for. It must be an integer.
            data: The data of the SED as a two dimensional array-like object.
                The first dimension has size same as the number of the knots and
                the second dimension has always size equal to two, with the
                first element representing the wavelength and the second the
                energy value.
        
        Raises:
            IdMismatchException: If the given ID is not in the reference sample
            AlreadySetException: If the SED data are aready set for the given ID
            InvalidPositionException: If the SED of the previous object is not yet set
            InvalidDimensionsException: If the given data dimensions are wrong
            InvalidAxisException: If there are decreasing wavelength values
        """
        
        sed_pos, _ = self.__index.getPositions(obj_id)
        if sed_pos != -1:
            raise AlreadySetException('SED for ID ' + str(obj_id) + ' is already set')
        
        if obj_id != self.firstIdMissingSedData():
            raise InvalidPositionException('SEDs of previous objects are missing')
        
        new_pos = self.__sed.appendSed(obj_id, data)
        self.__index.setSedPosition(obj_id, new_pos)
        
    
    def firstIdMissingSedData(self):
        """Returns the ID of the first reference sample object which does not
        have SED data already set, or None if all SEDs are set"""
        return self.__index.firstIdMissingSedData()
    
    
    def addPdzData(self, obj_id, data):
        """Adds the PDZ data of a reference sample object.
        
        Args:
            obj_id: The ID of the object to add the PDZ for. It must be an integer.
            data: The data of the PDZ as a two dimensional array-like object.
                The first dimension has size same as the number of the knots and
                the second dimension has always size equal to two, with the
                first element representing the wavelength and the second the
                probability value.
                
        Raises:
            IdMismatchException: If the given ID is not in the reference sample
            AlreadySetException: If the PDZ data are aready set for the given ID
            InvalidPositionException: If the PDZ of the previous object is not yet set
            InvalidDimensionsException: If the given data dimensions are wrong
            InvalidAxisException: If the wavelength values are not strictly increasing
            InvalidAxisException: If the wavelength values given are not matching
                the wavelength values of the other PDZs in the sample
        """
        
        _, pdz_pos = self.__index.getPositions(obj_id)
        if pdz_pos != -1:
            raise AlreadySetException('PDZ for ID ' + str(obj_id) + ' is already set')
        
        if obj_id != self.firstIdMissingPdzData():
            raise InvalidPositionException('PDZs of previous objects are missing')
        
        # Convert the data to a numpy array for easier handling
        data_arr = np.asarray(data, dtype=np.float32)
        if len(data_arr.shape) != 2 or data_arr.shape[1] != 2:
            raise InvalidDimensionsException()
        
        # Handle the wavelength axis
        existing_zs = self.__pdz.getRedshiftBins()
        if existing_zs is None:
            self.__pdz.setRedshiftBins(data_arr[:,0])
        else:
            if not np.array_equal(data_arr[:,0], existing_zs):
                raise InvalidAxisException('Given wavelengths are different than existing ones')
        
        new_pos = self.__pdz.appendPdz(obj_id, data_arr[:,1])
        self.__index.setPdzPosition(obj_id, new_pos)
        
    
    def firstIdMissingPdzData(self):
        """Returns the ID of the first reference sample object which does not
        have PDZ data already set, or None if all PDZs are set"""
        return self.__index.firstIdMissingPdzData()
    
    
    def iterate(self):
        """Returns an iterable object over the reference sample objects.
        
        The objects iterated provide the following members:
        - id: The ID of the object
        - sed: None if the SED is not set for the given object, otherwise the
            data of the SED as a two dimensional numpy array of single
            precission floats. The first dimension has size same as the number
            of the knots and the second dimension has always size equal to two,
            with the first element representing the wavelength and the second
            the energy value.
        - pdz: None if the PDZ is not set for the given object, otherwise the
            data of the PDZ as a two dimensional numpy array of single
            precission floats. The first dimension has size same as the number
            of the knots and the second dimension has always size equal to two,
            with the first element representing the wavelength and the second
            the probability value.
        """
        
        class Element(object):
            
            def __init__(self, obj_id, ref_sample):
                self.id = obj_id
                self.__ref_sample = ref_sample
                
            @property
            def sed(self):
                return self.__ref_sample.getSedData(self.id)
                
            @property
            def pdz(self):
                return self.__ref_sample.getPdzData(self.id)
        
        return (Element(i, self) for i in self.getIds())