"""
Created on: 10/11/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import pytest
import os
import numpy as np

from nnpz.exceptions import *
from nnpz.reference_sample.IndexProvider import IndexProvider

from tests.util_fixtures import temp_dir_fixture

###############################################################################

def test_constructor_missingFile(temp_dir_fixture):
    """Tests the case that the file does not exist"""
    
    # Given
    filename = os.path.join(temp_dir_fixture, 'missing')
    
    # Then
    with pytest.raises(FileNotFoundException):
        IndexProvider(filename)

###############################################################################

def test_constructor_wrongFileSize(temp_dir_fixture):
    """Tests the case that the file has wrong size"""
    
    # Given
    filename = os.path.join(temp_dir_fixture, 'wrong_size.bin')
    with open(filename, 'wb') as f:
        np.asarray([1,2,3,4], dtype=np.int64).tofile(f)
    
    # Then
    with pytest.raises(CorruptedFileException):
        IndexProvider(filename)
        
###############################################################################

def test_constructor_idMismatch(temp_dir_fixture):
    """Tests the case of a duplicate ID"""
    
    # Given
    filename = os.path.join(temp_dir_fixture, 'wrong_size.bin')
    with open(filename, 'wb') as f:
        np.asarray([1,-1,-1,
                    2,-1,-1,
                    3,-1,-1,
                    2,-1,-1], dtype=np.int64).tofile(f)
    
    # Then
    with pytest.raises(IdMismatchException):
        IndexProvider(filename)
        
###############################################################################

def test_constructor_nonIncreasingSedPos(temp_dir_fixture):
    """Tests the case where the SED position is not strictly increasing"""
    
    # Given
    filename = os.path.join(temp_dir_fixture, 'non_increasing_sed_pos.bin')
    with open(filename, 'wb') as f:
        np.asarray([1,1,-1,
                    2,2,-1,
                    3,1,-1,
                    4,-1,-1], dtype=np.int64).tofile(f)
    
    # Then
    with pytest.raises(InvalidPositionException):
        IndexProvider(filename)
        
###############################################################################

def test_constructor_nonIncreasingPdzPos(temp_dir_fixture):
    """Tests the case where the PDZ position is not strictly increasing"""
    
    # Given
    filename = os.path.join(temp_dir_fixture, 'non_increasing_pdz_pos.bin')
    with open(filename, 'wb') as f:
        np.asarray([1,-1,1,
                    2,-1,2,
                    3,-1,1,
                    4,-1,-1], dtype=np.int64).tofile(f)
    
    # Then
    with pytest.raises(InvalidPositionException):
        IndexProvider(filename)
        
###############################################################################

def test_constructor_negativeSedPos(temp_dir_fixture):
    """Tests the case where a SED position is negative"""
    
    # Given
    filename = os.path.join(temp_dir_fixture, 'negative_sed_pos.bin')
    with open(filename, 'wb') as f:
        np.asarray([1,1,1,
                    2,2,2,
                    3,-3,3,
                    4,4,4], dtype=np.int64).tofile(f)
    
    # Then
    with pytest.raises(InvalidPositionException):
        IndexProvider(filename)
        
###############################################################################

def test_constructor_negativePdzPos(temp_dir_fixture):
    """Tests the case where a PDZ position is negative"""
    
    # Given
    filename = os.path.join(temp_dir_fixture, 'negative_pdz_pos.bin')
    with open(filename, 'wb') as f:
        np.asarray([1,1,1,
                    2,2,2,
                    3,3,-3,
                    4,4,4], dtype=np.int64).tofile(f)
    
    # Then
    with pytest.raises(InvalidPositionException):
        IndexProvider(filename)
        
###############################################################################

def test_constructor_extraPositiveSedPos(temp_dir_fixture):
    """Tests the case where a SED position is positive after a -1"""
    
    # Given
    filename = os.path.join(temp_dir_fixture, 'extra_positive_sed_pos.bin')
    with open(filename, 'wb') as f:
        np.asarray([1,1,1,
                    2,2,2,
                    3,-1,3,
                    4,4,4], dtype=np.int64).tofile(f)
    
    # Then
    with pytest.raises(InvalidPositionException):
        IndexProvider(filename)
        
###############################################################################

def test_constructor_extraPositivePdzPos(temp_dir_fixture):
    """Tests the case where a PDZ position is positive after a -1"""
    
    # Given
    filename = os.path.join(temp_dir_fixture, 'extra_positive_pdz_pos.bin')
    with open(filename, 'wb') as f:
        np.asarray([1,1,1,
                    2,2,2,
                    3,3,-1,
                    4,4,4], dtype=np.int64).tofile(f)
    
    # Then
    with pytest.raises(InvalidPositionException):
        IndexProvider(filename)
        
###############################################################################

def test_getIdList(temp_dir_fixture):
    """Tests the getIdList() returns the correct IDs"""
    
    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        np.asarray([1,-1,-1,
                    2,-1,-1,
                    3,-1,-1,
                    4,-1,-1], dtype=np.int64).tofile(f)
    
    # When
    provider = IndexProvider(filename)
    id_list = provider.getIdList()
    
    # Then
    assert np.array_equal(id_list, [1,2,3,4])
        
###############################################################################

def test_getSedPositionList(temp_dir_fixture):
    """Tests the getSedPositionList() returns the correct positions"""
    
    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        np.asarray([1,10,-1,
                    2,20,-1,
                    3,30,-1,
                    4,-1,-1], dtype=np.int64).tofile(f)
    
    # When
    provider = IndexProvider(filename)
    pos_list = provider.getSedPositionList()
    
    # Then
    assert np.array_equal(pos_list, [10,20,30,-1])
        
###############################################################################

def test_getPdzPositionList(temp_dir_fixture):
    """Tests the getPdzPositionList() returns the correct positions"""
    
    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        np.asarray([1,-1,10,
                    2,-1,20,
                    3,-1,30,
                    4,-1,-1], dtype=np.int64).tofile(f)
    
    # When
    provider = IndexProvider(filename)
    pos_list = provider.getPdzPositionList()
    
    # Then
    assert np.array_equal(pos_list, [10,20,30,-1])
        
###############################################################################

def test_getPositions_success(temp_dir_fixture):
    """Tests the getPosittions() returns the correct positions for all IDs"""
    
    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        np.asarray([1,5,10,
                    2,15,20,
                    3,25,30,
                    4,-1,-1], dtype=np.int64).tofile(f)
    provider = IndexProvider(filename)
    
    for obj_id, sed_pos, pdz_pos in zip([1,2,3,4], [5,15,25,-1], [10,20,30,-1]):
        
        # When
        pos = provider.getPositions(obj_id)
    
        # Then
        assert pos[0] == sed_pos
        assert pos[1] == pdz_pos
        
###############################################################################

def test_getPositions_idMismatch(temp_dir_fixture):
    """Tests the getPosittions() with unknown ID"""
    
    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        np.asarray([1,5,10,
                    2,15,20,
                    3,25,30,
                    4,-1,-1], dtype=np.int64).tofile(f)
                    
    # When
    provider = IndexProvider(filename)
    
    # Then
    with pytest.raises(IdMismatchException):
        provider.getPositions(5)
        
###############################################################################

def test_appendId_duplicateId(temp_dir_fixture):
    """Tests the appendId() with an ID already in the index"""
    
    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        np.asarray([1,5,10,
                    2,15,20,
                    3,25,30,
                    4,-1,-1], dtype=np.int64).tofile(f)
                    
    # When
    provider = IndexProvider(filename)
    
    # Then
    with pytest.raises(DuplicateIdException):
        provider.appendId(3)
        
###############################################################################

def test_appendId_success(temp_dir_fixture):
    """Tests successful call of appendId()"""
    
    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        np.asarray([1,5,10,
                    2,15,20,
                    3,25,30,
                    4,-1,-1], dtype=np.int64).tofile(f)
    old_size = os.path.getsize(filename)
                    
    # When
    provider = IndexProvider(filename)
    provider.appendId(5)
    pos = provider.getPositions(5)
    
    # Then
    assert pos[0] == -1
    assert pos[1] == -1
    assert os.path.getsize(filename) == old_size + 3 * 8
    with open(filename, 'rb') as f:
        f.seek(old_size)
        from_file = np.fromfile(f, count=3, dtype=np.int64)
        assert np.array_equal(from_file, [5, -1, -1])
        
###############################################################################

def test_setSedPosition_idMismatch(temp_dir_fixture):
    """Tests the setSedPosition() with unknown ID"""
    
    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        np.asarray([1,5,10,
                    2,15,20,
                    3,25,30,
                    4,-1,-1], dtype=np.int64).tofile(f)
                    
    # When
    provider = IndexProvider(filename)
    
    # Then
    with pytest.raises(IdMismatchException):
        provider.setSedPosition(5, 30)
        
###############################################################################

def test_setSedPosition_alreadySet(temp_dir_fixture):
    """Tests the setSedPosition() for the case the position is already set"""
    
    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        np.asarray([1,5,10,
                    2,15,20,
                    3,25,30,
                    4,-1,-1], dtype=np.int64).tofile(f)
                    
    # When
    provider = IndexProvider(filename)
    
    # Then
    with pytest.raises(AlreadySetException):
        provider.setSedPosition(3, 30)
        
###############################################################################

def test_setSedPosition_invalidPosition(temp_dir_fixture):
    """Tests the setSedPosition() for the invalid position value"""
    
    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        np.asarray([1,5,10,
                    2,15,20,
                    3,25,30,
                    4,-1,-1,
                    5,-1,-1], dtype=np.int64).tofile(f)
                    
    # When
    provider = IndexProvider(filename)
    
    # Then
    with pytest.raises(InvalidPositionException):
        provider.setSedPosition(4, 24)  # Smaller than previous
    with pytest.raises(InvalidPositionException):
        provider.setSedPosition(4, 25)  # Same with previous
    with pytest.raises(InvalidPositionException):
        provider.setSedPosition(5, 35)  # Previous not set
        
        
###############################################################################

def test_setSedPosition_success(temp_dir_fixture):
    """Test successful call of the setSedPosition()"""
    
    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        np.asarray([1,5,10,
                    2,15,20,
                    3,25,30,
                    4,-1,-1,
                    5,-1,-1], dtype=np.int64).tofile(f)
                    
    # When
    expected = 35
    provider = IndexProvider(filename)
    provider.setSedPosition(4, expected)
    pos = provider.getPositions(4)
    
    # Then
    assert pos[0] == expected
    with open(filename, 'rb') as f:
        f.seek(3 * (3 * 8) + 8)
        from_file = np.fromfile(f, count=1, dtype=np.int64)[0]
        assert from_file == expected
        
###############################################################################

def test_setPdzPosition_idMismatch(temp_dir_fixture):
    """Tests the setPdzPosition() with unknown ID"""
    
    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        np.asarray([1,5,10,
                    2,15,20,
                    3,25,30,
                    4,-1,-1], dtype=np.int64).tofile(f)
                    
    # When
    provider = IndexProvider(filename)
    
    # Then
    with pytest.raises(IdMismatchException):
        provider.setPdzPosition(5, 40)
        
###############################################################################

def test_setPdzPosition_alreadySet(temp_dir_fixture):
    """Tests the setPdzPosition() for the case the position is already set"""
    
    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        np.asarray([1,5,10,
                    2,15,20,
                    3,25,30,
                    4,-1,-1], dtype=np.int64).tofile(f)
                    
    # When
    provider = IndexProvider(filename)
    
    # Then
    with pytest.raises(AlreadySetException):
        provider.setPdzPosition(3, 40)
        
###############################################################################

def test_setdzPosition_invalidPosition(temp_dir_fixture):
    """Tests the setPdzPosition() for the invalid position value"""
    
    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        np.asarray([1,5,10,
                    2,15,20,
                    3,25,30,
                    4,-1,-1,
                    5,-1,-1], dtype=np.int64).tofile(f)
                    
    # When
    provider = IndexProvider(filename)
    
    # Then
    with pytest.raises(InvalidPositionException):
        provider.setPdzPosition(4, 29)  # Smaller than previous
    with pytest.raises(InvalidPositionException):
        provider.setPdzPosition(4, 30)  # Same with previous
    with pytest.raises(InvalidPositionException):
        provider.setPdzPosition(5, 40)  # Previous not set
        
        
###############################################################################

def test_setPdzPosition_success(temp_dir_fixture):
    """Test successful call of the setPdzPosition()"""
    
    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        np.asarray([1,5,10,
                    2,15,20,
                    3,25,30,
                    4,-1,-1,
                    5,-1,-1], dtype=np.int64).tofile(f)
    expected = 40
                    
    # When
    provider = IndexProvider(filename)
    provider.setPdzPosition(4, expected)
    pos = provider.getPositions(4)
    
    # Then
    assert pos[1] == expected
    with open(filename, 'rb') as f:
        f.seek(3 * (3 * 8) + (2 * 8))
        from_file = np.fromfile(f, count=1, dtype=np.int64)[0]
        assert from_file == expected
        
###############################################################################

