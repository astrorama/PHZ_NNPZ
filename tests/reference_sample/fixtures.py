"""
Created on: 10/11/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import os
import pytest
import numpy as np
from astropy.table import Table, Column

from tests.util_fixtures import temp_dir_fixture

##############################################################################

@pytest.fixture()
def sed_list_fixture():
    """Returns a list of SEDs to be used for testing.
    
    The SEDs are the following:
      ID    Data
     ----  --------------
      1    [(1., 100.), (1.5, 168.3), (1.7,96.12)]
      3    [(167.7,4.2), (456.49,7.8)]
     12    [(1.,1.), (2.,2.), (3.,3.), (4.,4.),(5.,5.)]
      7    [(56.,78.1)]
    """
    return [
        [1, np.asarray([(1., 100.), (1.5, 168.3), (1.7,96.12)], dtype=np.float32)],
        [3, np.asarray([(167.7,4.2), (456.49,7.8)], dtype=np.float32)],
        [12, np.asarray([(1.,1.), (2.,2.), (3.,3.), (4.,4.),(5.,5.)], dtype=np.float32)],
        [7, np.asarray([(56.,78.1)], dtype=np.float32)]
    ]

##############################################################################

@pytest.fixture()
def sed_data_file_fixture(temp_dir_fixture, sed_list_fixture):
    """Creates a SED data file to be used for testing.
    
    Returns: The path the the newly created file.
    
    The created file contains the SEDs of the sed_list_fixture.
    """
    
    filename = os.path.join(temp_dir_fixture, 'sed_data.bin')
    with open(filename, 'wb') as out:
        for i, data in sed_list_fixture:
            # Write the index
            np.asarray([i], dtype='int64').tofile(out)
            # Write the length
            np.asarray([len(data)], dtype='uint32').tofile(out)
            # Write the data
            data_arr = np.asarray(data, dtype='float32')
            data_arr.flatten().tofile(out)
    
    return filename
    
##############################################################################

@pytest.fixture()
def redshift_bins_fixture():
    """Returns an array with the redsift bins of the PDZs to be used for testing"""
    return np.asarray([1,2,5,6,8,9], dtype=np.float32)
    
##############################################################################

@pytest.fixture()
def pdz_list_fixture():
    """Returns a list of PDZs to be used for testing."""
    return [
        [1, np.asarray([1,2,3,4,5,6], dtype=np.float32)],
        [3, np.asarray([2,3,4,5,6,7], dtype=np.float32)],
        [12, np.asarray([3,4,5,6,7,8], dtype=np.float32)],
        [7, np.asarray([4,5,6,7,8,9], dtype=np.float32)]
    ]
    
##############################################################################

@pytest.fixture()
def pdz_data_file_fixture(temp_dir_fixture, redshift_bins_fixture, pdz_list_fixture):
    """Creates a PDZ data file to be used for testing.
    
    Returns: The path to the newly created file.
    
    The created file contains the PDZs of the pdz_list_fixture and the redshift
    bins of the redshift_bins_fixture.
    """
    
    filename = os.path.join(temp_dir_fixture, 'pdz_data.bin')
    with open(filename, 'wb') as out:
        # Write the length of each PDZ
        np.asarray([len(redshift_bins_fixture)], dtype=np.uint32).tofile(out)
        # Write the redshift bins
        np.asarray(redshift_bins_fixture, dtype=np.float32).tofile(out)
        #Write all the PDZs
        for i, data in pdz_list_fixture:
            np.asarray([i], dtype=np.int64).tofile(out)
            np.asarray(data, dtype=np.float32).tofile(out)
            
    return filename
    
##############################################################################
    
@pytest.fixture()
def photometry_data_fixture():
    """Returns data for two photometry files.
    
    The data are the following:
    
    File photo1.fits:
    ID A1 A2 A3 A4
    -- -- -- -- --
     1  1  5  9 13
     3  2  6 10 14
    12  3  7 11 15
     7  4  8 12 16
     
    File photo2.fits:
    ID B1 B2 B3 A4
    -- -- -- -- --
     1 17 21 25 29
     3 18 22 26 30
    12 19 23 27 31
     7 20 24 28 32
    """
    return {
        'photo1.fits': {
            'ID': [1,3,12,7],
            'A1': [1,2,3,4],
            'A2': [5,6,7,8],
            'A3': [9,10,11,12],
            'A4': [13,14,15,16]
        },
        'photo2.fits': {
            'ID': [1,3,12,7],
            'B1': [17,18,19,20],
            'B2': [21,22,23,24],
            'B3': [25,26,27,28],
            'A4': [29,30,31,32]
        }
    }
    
##############################################################################
    
@pytest.fixture()
def photometry_dir_fixture(temp_dir_fixture, photometry_data_fixture):
    """Returns a directory which contains FITS files with photometry data"""
    for f in photometry_data_fixture:
        columns = photometry_data_fixture[f]
        t = Table()
        t.meta['EXTNAME'] = 'NNPZ_PHOTOMETRY'
        t['ID'] = Column(np.asarray(columns['ID'], dtype=np.int64))
        for name in columns:
            data = columns[name]
            if name != 'ID':
                t[name] = Column(np.asarray(data, dtype=np.float32))
        t.write(os.path.join(temp_dir_fixture, f), format='fits')
    return temp_dir_fixture
    
##############################################################################
    
@pytest.fixture()
def reference_sample_dir_fixture(temp_dir_fixture, sed_data_file_fixture,
                                 pdz_data_file_fixture, photometry_dir_fixture,
                                 sed_list_fixture, redshift_bins_fixture):
    """Returns a directory which contains a reference sample"""
    
    # Create the index file
    pdz_length = len(redshift_bins_fixture)
    pdz_offset = 4 + 4 * pdz_length
    with open(os.path.join(temp_dir_fixture, 'index.bin'), 'wb') as f:
        
        # Add all the entries for the objects with data
        sed_pos = 0
        pdz_pos = pdz_offset
        for obj_id, sed_data in sed_list_fixture:
            np.asarray([obj_id, sed_pos, pdz_pos], dtype=np.int64).tofile(f)
            sed_pos += 8 + 4 + 4 * sed_data.size
            pdz_pos += 8 + 4 * pdz_length
        
        # Add two more objects without data
        np.asarray([100, -1, -1], dtype=np.int64).tofile(f)
        np.asarray([101, -1, -1], dtype=np.int64).tofile(f)
    
    return temp_dir_fixture
    
##############################################################################