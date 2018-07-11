"""
Created on: 10/11/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import os
import pytest
import numpy as np

from ..util_fixtures import temp_dir_fixture

##############################################################################

@pytest.fixture()
def sed_list_fixture():
    """Returns a list of SEDs to be used for testing.
    
    The SEDs are the following:
      File   ID  Data
     ------ ---- --------------
       1     1   [(1., 100.), (1.5, 168.3), (1.7,96.12)]
       1     3   [(167.7,4.2), (456.49,7.8)]
       1    12   [(1.,1.), (2.,2.), (3.,3.), (4.,4.),(5.,5.)]
       1     7   [(56.,78.1)]
       2    34   [(6.,6.), (7.,7.)]
       2    56   [(8.,8.), (9.,9.)]
    """
    return {
        1:[
            [1, np.asarray([(1., 100.), (1.5, 168.3), (1.7,96.12)], dtype=np.float32)],
            [3, np.asarray([(167.7,4.2), (456.49,7.8)], dtype=np.float32)],
            [12, np.asarray([(1.,1.), (2.,2.), (3.,3.), (4.,4.),(5.,5.)], dtype=np.float32)],
            [7, np.asarray([(56.,78.1)], dtype=np.float32)]
        ],
        2:[
            [34, np.asarray([(6.,6.), (7.,7.)], dtype=np.float32)],
            [56, np.asarray([(8.,8.), (9.,9.)], dtype=np.float32)]
        ]
    }

##############################################################################

@pytest.fixture()
def sed_data_files_fixture(temp_dir_fixture, sed_list_fixture):
    """Creates SED data files to be used for testing.

    Returns: A map with keys the file indices and values the paths of the files
    
    The created files contains the SEDs of the sed_list_fixture.
    """

    result = {}
    for file_index in sed_list_fixture:
        filename = os.path.join(temp_dir_fixture, 'sed_data_{}.bin'.format(file_index))
        with open(filename, 'wb') as out:
            for i, data in sed_list_fixture[file_index]:
                # Write the index
                np.asarray([i], dtype='int64').tofile(out)
                # Write the length
                np.asarray([len(data)], dtype='uint32').tofile(out)
                # Write the data
                data_arr = np.asarray(data, dtype='float32')
                data_arr.flatten().tofile(out)
        result[file_index] = filename
    
    return result
    
##############################################################################

@pytest.fixture()
def redshift_bins_fixture():
    """Returns an array with the redsift bins of the PDZs to be used for testing"""
    return np.asarray([1,2,5,6,8,9], dtype=np.float32)
    
##############################################################################

@pytest.fixture()
def pdz_list_fixture():
    """Returns a list of PDZs to be used for testing."""
    return {
        1: [
            [1, np.asarray([1,2,3,4,5,6], dtype=np.float32)],
            [3, np.asarray([2,3,4,5,6,7], dtype=np.float32)],
            [12, np.asarray([3,4,5,6,7,8], dtype=np.float32)],
            [7, np.asarray([4,5,6,7,8,9], dtype=np.float32)]
        ],
        2: [
            [34, np.asarray([11,12,13,14,15,16], dtype=np.float32)],
            [56, np.asarray([12,13,14,15,16,17], dtype=np.float32)]
        ]
    }
    
##############################################################################

@pytest.fixture()
def pdz_data_files_fixture(temp_dir_fixture, redshift_bins_fixture, pdz_list_fixture):
    """Creates PDZ data files to be used for testing.
    
    Returns: A map with keys the file indices and values he path to the newly created files
    
    The created files contain the PDZs of the pdz_list_fixture and the redshift
    bins of the redshift_bins_fixture.
    """

    result = {}
    for file_index in pdz_list_fixture:
        filename = os.path.join(temp_dir_fixture, 'pdz_data_{}.bin'.format(file_index))
        with open(filename, 'wb') as out:
            # Write the length of each PDZ
            np.asarray([len(redshift_bins_fixture)], dtype=np.uint32).tofile(out)
            # Write the redshift bins
            np.asarray(redshift_bins_fixture, dtype=np.float32).tofile(out)
            #Write all the PDZs
            for i, data in pdz_list_fixture[file_index]:
                np.asarray([i], dtype=np.int64).tofile(out)
                np.asarray(data, dtype=np.float32).tofile(out)
        result[file_index] = filename
            
    return result
    
##############################################################################
    
@pytest.fixture()
def reference_sample_dir_fixture(temp_dir_fixture, sed_data_files_fixture,
                                 pdz_data_files_fixture, sed_list_fixture,
                                 redshift_bins_fixture):
    """Returns a directory which contains a reference sample"""
    
    # Create the index file
    pdz_length = len(redshift_bins_fixture)
    pdz_offset = 4 + 4 * pdz_length
    with open(os.path.join(temp_dir_fixture, 'index.bin'), 'wb') as f:

        for file_index in sed_list_fixture:
            sed_pos = 0
            pdz_pos = pdz_offset
            for obj_id, sed_data in sed_list_fixture[file_index]:
                np.asarray([obj_id], dtype=np.int64).tofile(f)
                np.asarray([file_index], dtype=np.uint16).tofile(f)
                np.asarray([sed_pos], dtype=np.int64).tofile(f)
                np.asarray([file_index], dtype=np.uint16).tofile(f)
                np.asarray([pdz_pos], dtype=np.int64).tofile(f)
                sed_pos += 8 + 4 + 4 * sed_data.size
                pdz_pos += 8 + 4 * pdz_length
        
        # Add two more objects without data
        np.asarray([100], dtype=np.int64).tofile(f)
        np.asarray([0], dtype=np.uint16).tofile(f)
        np.asarray([-1], dtype=np.int64).tofile(f)
        np.asarray([0], dtype=np.uint16).tofile(f)
        np.asarray([-1], dtype=np.int64).tofile(f)
        np.asarray([101], dtype=np.int64).tofile(f)
        np.asarray([0], dtype=np.uint16).tofile(f)
        np.asarray([-1], dtype=np.int64).tofile(f)
        np.asarray([0], dtype=np.uint16).tofile(f)
        np.asarray([-1], dtype=np.int64).tofile(f)
    
    return temp_dir_fixture
    
##############################################################################
