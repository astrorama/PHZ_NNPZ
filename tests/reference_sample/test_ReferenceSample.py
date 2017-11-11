"""
Created on: 10/11/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import pytest
import os
from astropy.table import Table
import numpy as np

from nnpz import ReferenceSample

from tests.util_fixtures import temp_dir_fixture

###############################################################################

def test_createNew_dirExists(temp_dir_fixture):
    '''Tests that if the directory exists an exception is raised'''
    
    # Given
    dir_name = os.path.join(temp_dir_fixture, 'ref_sample')
    os.makedirs(dir_name)
    
    # Then
    with pytest.raises(OSError):
        ReferenceSample.createNew(dir_name)
    
###############################################################################

def test_createNew_success(temp_dir_fixture):
    '''Tests that all the files of the reference sample are constructed correctly'''
    
    # Given
    dir_name = os.path.join(temp_dir_fixture, 'ref_sample')
    
    # When
    ReferenceSample.createNew(dir_name)
    
    # Then
    
    # Check the directory was creted correctly
    assert os.path.isdir(dir_name)
    
    # Check that the index file is created and that it contains the ID, SED_POS
    # and PDZ_POS columns with the correct types
    index_file = os.path.join(dir_name, 'index.fits')
    assert os.path.exists(index_file)
    table = Table.read(index_file)
    assert 'ID' in table.colnames
    assert table.dtype['ID'].type == np.int64
    assert 'SED_POS' in table.colnames
    assert table.dtype['SED_POS'].type == np.int64
    assert 'PDZ_POS' in table.colnames
    assert table.dtype['SED_POS'].type == np.int64
    
    # Check that the empty SED file is created
    sed_file = os.path.join(dir_name, 'sed_data.bin')
    assert os.path.exists(sed_file)
    assert os.path.getsize(sed_file) == 0
    
    # Check that the empty PDZ file is created
    pdz_file = os.path.join(dir_name, 'pdz_data.bin')
    assert os.path.exists(pdz_file)
    assert os.path.getsize(pdz_file) == 0
    
###############################################################################
    