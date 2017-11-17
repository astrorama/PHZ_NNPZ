"""
Created on: 10/11/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import pytest
import os
import numpy as np

from nnpz.exceptions import *
from nnpz.reference_sample.SedDataProvider import SedDataProvider

from tests.util_fixtures import temp_dir_fixture
from .fixtures import sed_data_file_fixture, sed_list_fixture

###############################################################################

def test_constructor_missingFile(temp_dir_fixture):
    """Tests the case that the file does not exist"""
    
    # Given
    filename = os.path.join(temp_dir_fixture, 'missing')
    
    # Then
    with pytest.raises(FileNotFoundException):
        SedDataProvider(filename)

###############################################################################

def test_readSed_success(sed_data_file_fixture, sed_list_fixture):
    """Tests successful call of readSed()"""
    
    # Given
    provider = SedDataProvider(sed_data_file_fixture)
    
    pos = 0
    for expected_id, expected_data in sed_list_fixture:
        
        # When
        found_id, found_data = provider.readSed(pos)
        
        # Then
        assert found_id == expected_id
        assert len(found_data.shape) == 2
        assert found_data.shape[0] == len(expected_data)
        assert found_data.shape[1] == 2
        for i, (x, y) in enumerate(expected_data):
            assert found_data[i][0] == x
            assert found_data[i][1] == y
        
        # Increase the position for the ID, the length and the SED data
        pos = pos + 8 + 4 + (4 * 2 * len(expected_data))
    
###############################################################################

def test_appendSed_success(sed_data_file_fixture):
    """Tests successful call of appendSed()"""
    
    # Given
    expected_id = 87
    expected_pos = os.path.getsize(sed_data_file_fixture)
    expected_data = np.asarray([(1.,10,), (2.,20.), (3.,30)], dtype=np.float32)
    expected_size = expected_pos + 8 + 4 + (2 * 4 * len(expected_data))
    provider = SedDataProvider(sed_data_file_fixture)
    
    # When
    pos = provider.appendSed(expected_id, expected_data)
    file_size = os.path.getsize(sed_data_file_fixture)
    
    # Then
    assert file_size == expected_size
    assert pos == expected_pos
    with open(sed_data_file_fixture, 'rb') as f:
        f.seek(expected_pos)
        # Check that the ID is correct
        file_id = np.fromfile(f, count=1, dtype='int64')[0]
        assert file_id == expected_id
        # Check that the length is correct
        file_length = np.fromfile(f, count=1, dtype='uint32')[0]
        assert file_length == len(expected_data)
        # Check that the data are written correctly
        file_data = np.fromfile(f, count=2*file_length, dtype='float32')
        assert np.array_equal(file_data, expected_data.flatten())
    
###############################################################################

def test_appendSed_invalidDataDimension(sed_data_file_fixture):
    """Test that the appendSed() method raises an exception for wrong data dimensions"""
    
    # Given
    sed_id = 87
    three_dim_data = np.zeros((100,2,2), dtype=np.float32)
    wrong_second_dimension = np.zeros((100,3), dtype=np.float32)
    
    
    # When
    provider = SedDataProvider(sed_data_file_fixture)
    
    # Then
    with pytest.raises(InvalidDimensionsException):
        provider.appendSed(sed_id, three_dim_data)
    with pytest.raises(InvalidDimensionsException):
        provider.appendSed(sed_id, wrong_second_dimension)
    
###############################################################################

def test_appendSed_nonIncreasingWavelength(sed_data_file_fixture):
    """Test that the appendSed() method raises an exception for non increasing wavelengths"""
    
    # Given
    sed_id = 87
    wrong_wavelength = np.asarray([(1.,10,), (3.,20.), (2.,30)], dtype=np.float32)
    
    
    # When
    provider = SedDataProvider(sed_data_file_fixture)
    
    # Then
    with pytest.raises(InvalidAxisException):
        provider.appendSed(sed_id, wrong_wavelength)

###############################################################################

def test_validate_diffInputSize(sed_data_file_fixture):
    """Test that the validate() raises an exception if the id and pos lists have different size"""
    
    # Given
    id_list = [1, 2, 3]
    pos_list = [1, 2, 3, 4]
    
    # When
    provider = SedDataProvider(sed_data_file_fixture)
    
    # Then
    with pytest.raises(InvalidDimensionsException):
        provider.validate(id_list, pos_list)

###############################################################################

def test_validate_consistentFile(sed_data_file_fixture, sed_list_fixture):
    """Test that when the file is consistent with the given lists it returns None"""
    
    # Given
    id_list = [x for x,_ in sed_list_fixture]
    pos_list = [0]
    for i in range(1, len(id_list)):
        pos_list.append(pos_list[i-1] + 8 + 4 + (4 * 2 * len(sed_list_fixture[i-1][1])))
    provider = SedDataProvider(sed_data_file_fixture)
    
    # When
    message = provider.validate(id_list, pos_list)
    
    # Then
    assert message is None

###############################################################################

def test_validate_incosistentId(sed_data_file_fixture, sed_list_fixture):
    """Test the case where an ID differs"""
    
    # Given
    id_list = [x for x,_ in sed_list_fixture]
    old_id = id_list[1]
    id_list[1] += 1
    pos_list = [0]
    for i in range(1, len(id_list)):
        pos_list.append(pos_list[i-1] + 8 + 4 + (4 * 2 * len(sed_list_fixture[i-1][1])))
    provider = SedDataProvider(sed_data_file_fixture)
    
    # When
    message = provider.validate(id_list, pos_list)
    
    # Then
    assert message == 'Incosistent IDs (' + str(id_list[1]) + ', ' + str(old_id) + ')'

###############################################################################

def test_validate_incosistentPosition(sed_data_file_fixture, sed_list_fixture):
    """Test the case where a position is incosistent"""
    
    # Given
    id_list = [x for x,_ in sed_list_fixture]
    pos_list = [0]
    for i in range(1, len(id_list)):
        pos_list.append(pos_list[i-1] + 8 + 4 + (4 * 2 * len(sed_list_fixture[i-1][1])))
    old_pos = pos_list[1]
    pos_list[1] += 1
    provider = SedDataProvider(sed_data_file_fixture)
    
    # When
    message = provider.validate(id_list, pos_list)
    
    # Then
    assert message == 'Incosistent position for ID=' + str(id_list[1]) + ' (' + str(pos_list[1]) + ', ' + str(old_pos) + ')'

###############################################################################

def test_validate_incosistentSedLength(sed_data_file_fixture, sed_list_fixture):
    """Test the case where a length is bigger than the file size"""
    
    # Given
    id_list = [x for x,_ in sed_list_fixture]
    pos_list = [0]
    for i in range(1, len(id_list)):
        pos_list.append(pos_list[i-1] + 8 + 4 + (4 * 2 * len(sed_list_fixture[i-1][1])))
    provider = SedDataProvider(sed_data_file_fixture)
    
    # When
    with open(sed_data_file_fixture, 'rb+') as f:
        f.seek(pos_list[1] + 8)
        np.asarray([10000], dtype='uint32').tofile(f)
    message = provider.validate(id_list, pos_list)
    
    # Then
    assert message == 'Data length bigger than file for ID=' + str(id_list[1])

###############################################################################

def test_validate_extraSedsInFile(sed_data_file_fixture, sed_list_fixture):
    """Test the case where the file contains extra SEDs"""
    
    # Given
    id_list = [x for x,_ in sed_list_fixture]
    pos_list = [0]
    for i in range(1, len(id_list)):
        pos_list.append(pos_list[i-1] + 8 + 4 + (4 * 2 * len(sed_list_fixture[i-1][1])))
    provider = SedDataProvider(sed_data_file_fixture)
    
    id_list = id_list[:-1]
    pos_list = pos_list[:-1]
    
    # When
    message = provider.validate(id_list, pos_list)
    
    # Then
    assert message == 'File contains extra SEDs'

###############################################################################

def test_validate_minusPositionConsistent(sed_data_file_fixture, sed_list_fixture):
    """Test the case where the file is consistent and the pos_list contains -1"""
    
    # Given
    id_list = [x for x,_ in sed_list_fixture]
    pos_list = [0]
    for i in range(1, len(id_list)):
        pos_list.append(pos_list[i-1] + 8 + 4 + (4 * 2 * len(sed_list_fixture[i-1][1])))
    provider = SedDataProvider(sed_data_file_fixture)
    
    id_list.append(100)
    pos_list.append(-1)
    id_list.append(101)
    pos_list.append(0)
    
    # When
    message = provider.validate(id_list, pos_list)
    
    # Then
    assert message is None

###############################################################################

def test_validate_extraSedsInFile(sed_data_file_fixture, sed_list_fixture):
    """Test the case where the file contains extra SEDs and the pos_list contains -1"""
    
    # Given
    id_list = [x for x,_ in sed_list_fixture]
    pos_list = [0]
    for i in range(1, len(id_list)):
        pos_list.append(pos_list[i-1] + 8 + 4 + (4 * 2 * len(sed_list_fixture[i-1][1])))
    provider = SedDataProvider(sed_data_file_fixture)
    
    pos_list[-2] = -1
    
    # When
    message = provider.validate(id_list, pos_list)
    
    # Then
    assert message == 'File contains extra SEDs'

###############################################################################

