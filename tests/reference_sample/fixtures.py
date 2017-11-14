"""
Created on: 10/11/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import os
import pytest
import numpy as np

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