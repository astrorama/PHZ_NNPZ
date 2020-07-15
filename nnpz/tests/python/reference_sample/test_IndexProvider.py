#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
#
# This library is free software; you can redistribute it and/or modify it under the terms of
# the GNU Lesser General Public License as published by the Free Software Foundation;
# either version 3.0 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this library;
# if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301 USA
#

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

from .fixtures import temp_dir_fixture


def _addEntryToFile(values, f):
    np.asarray([values[0]], dtype=np.int64).tofile(f)
    np.asarray([values[1]], dtype=np.uint16).tofile(f)
    np.asarray([values[2]], dtype=np.int64).tofile(f)
    np.asarray([values[3]], dtype=np.uint16).tofile(f)
    np.asarray([values[4]], dtype=np.int64).tofile(f)

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
        _addEntryToFile([1,0,-1,0,-1], f)
        _addEntryToFile([2,0,-1,0,-1], f)
        _addEntryToFile([3,0,-1,0,-1], f)
        _addEntryToFile([2,0,-1,0,-1], f)

    # Then
    with pytest.raises(IdMismatchException):
        IndexProvider(filename)

###############################################################################

def test_constructor_negativeSedPos(temp_dir_fixture):
    """Tests the case where a SED position is negative"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'negative_sed_pos.bin')
    with open(filename, 'wb') as f:
        _addEntryToFile([1,0,1,0,1], f)
        _addEntryToFile([2,0,2,0,2], f)
        _addEntryToFile([3,0,-3,0,3], f)
        _addEntryToFile([4,0,4,0,4], f)

    # Then
    with pytest.raises(InvalidPositionException):
        IndexProvider(filename)

###############################################################################

def test_constructor_negativePdzPos(temp_dir_fixture):
    """Tests the case where a PDZ position is negative"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'negative_pdz_pos.bin')
    with open(filename, 'wb') as f:
        _addEntryToFile([1,0,1,0,1], f)
        _addEntryToFile([2,0,2,0,2], f)
        _addEntryToFile([3,0,3,0,-3], f)
        _addEntryToFile([4,0,4,0,4], f)

    # Then
    with pytest.raises(InvalidPositionException):
        IndexProvider(filename)

###############################################################################

def test_getIdList(temp_dir_fixture):
    """Tests the getIdList() returns the correct IDs"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        _addEntryToFile([1,0,-1,0,-1], f)
        _addEntryToFile([2,0,-1,0,-1], f)
        _addEntryToFile([3,0,-1,0,-1], f)
        _addEntryToFile([4,0,-1,0,-1], f)

    # When
    provider = IndexProvider(filename)
    id_list = provider.getIdList()

    # Then
    assert np.array_equal(id_list, [1,2,3,4])

###############################################################################

def test_getSedFileList(temp_dir_fixture):
    """Tests the getSedFileList() returns the correct info"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        _addEntryToFile([1,1,10,0,-1], f)
        _addEntryToFile([2,1,20,0,-1], f)
        _addEntryToFile([3,2,30,0,-1], f)
        _addEntryToFile([4,0,-1,0,-1], f)

    # When
    provider = IndexProvider(filename)
    pos_list = provider.getSedFileList()

    # Then
    assert np.array_equal(pos_list, [1,1,2,0])

###############################################################################

def test_getSedPositionList(temp_dir_fixture):
    """Tests the getSedPositionList() returns the correct info"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        _addEntryToFile([1,1,10,0,-1], f)
        _addEntryToFile([2,1,20,0,-1], f)
        _addEntryToFile([3,2,30,0,-1], f)
        _addEntryToFile([4,0,-1,0,-1], f)

    # When
    provider = IndexProvider(filename)
    pos_list = provider.getSedPositionList()

    # Then
    assert np.array_equal(pos_list, [10,20,30,-1])

###############################################################################

def test_getPdzFileList(temp_dir_fixture):
    """Tests the getPdzFileList() returns the correct positions"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        _addEntryToFile([1,0,-1,1,10], f)
        _addEntryToFile([2,0,-1,1,20], f)
        _addEntryToFile([3,0,-1,2,30], f)
        _addEntryToFile([4,0,-1,0,-1], f)

    # When
    provider = IndexProvider(filename)
    pos_list = provider.getPdzFileList()

    # Then
    assert np.array_equal(pos_list, [1,1,2,0])

###############################################################################

def test_getPdzPositionList(temp_dir_fixture):
    """Tests the getPdzPositionList() returns the correct positions"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        _addEntryToFile([1,0,-1,1,10], f)
        _addEntryToFile([2,0,-1,1,20], f)
        _addEntryToFile([3,0,-1,2,30], f)
        _addEntryToFile([4,0,-1,0,-1], f)

    # When
    provider = IndexProvider(filename)
    pos_list = provider.getPdzPositionList()

    # Then
    assert np.array_equal(pos_list, [10,20,30,-1])

###############################################################################

def test_getFilesAndPositions_success(temp_dir_fixture):
    """Tests the getFilesAndPositions() returns the correct info for all IDs"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        _addEntryToFile([1,1,5,1,10], f)
        _addEntryToFile([2,1,15,2,20], f)
        _addEntryToFile([3,2,25,2,30], f)
        _addEntryToFile([4,0,-1,0,-1], f)
    provider = IndexProvider(filename)

    for obj_id, sed_file, sed_pos, pdz_file, pdz_pos in zip([1,2,3,4],
                                                            [1,1,2,0], [5,15,25,-1],
                                                            [1,2,2,0], [10,20,30,-1]):

        # When
        info = provider.getFilesAndPositions(obj_id)

        # Then
        assert info[0] == sed_file
        assert info[1] == sed_pos
        assert info[2] == pdz_file
        assert info[3] == pdz_pos

###############################################################################

def test_getgetFilesAndPositions_idMismatch(temp_dir_fixture):
    """Tests the getFilesAndPositions() with unknown ID"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        _addEntryToFile([1,1,5,1,10], f)
        _addEntryToFile([2,1,15,2,20], f)
        _addEntryToFile([3,2,25,2,30], f)
        _addEntryToFile([4,0,-1,0,-1], f)

    # When
    provider = IndexProvider(filename)

    # Then
    with pytest.raises(IdMismatchException):
        provider.getFilesAndPositions(5)

###############################################################################

def test_appendId_duplicateId(temp_dir_fixture):
    """Tests the appendId() with an ID already in the index"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        _addEntryToFile([1,1,5,1,10], f)
        _addEntryToFile([2,1,15,2,20], f)
        _addEntryToFile([3,2,25,2,30], f)
        _addEntryToFile([4,0,-1,0,-1], f)

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
        _addEntryToFile([1,1,5,1,10], f)
        _addEntryToFile([2,1,15,2,20], f)
        _addEntryToFile([3,2,25,2,30], f)
        _addEntryToFile([4,0,-1,0,-1], f)
    old_size = os.path.getsize(filename)

    # When
    provider = IndexProvider(filename)
    provider.appendId(5)
    info = provider.getFilesAndPositions(5)

    # Then
    assert info[0] == 0
    assert info[1] == -1
    assert info[2] == 0
    assert info[3] == -1
    assert os.path.getsize(filename) == old_size + 28
    with open(filename, 'rb') as f:
        f.seek(old_size)
        assert np.fromfile(f, count=1, dtype=np.int64)[0] == 5
        assert np.fromfile(f, count=1, dtype=np.uint16)[0] == 0
        assert np.fromfile(f, count=1, dtype=np.int64)[0] == -1
        assert np.fromfile(f, count=1, dtype=np.uint16)[0] == 0
        assert np.fromfile(f, count=1, dtype=np.int64)[0] == -1

###############################################################################

def test_setSedFileAndPosition_idMismatch(temp_dir_fixture):
    """Tests the setSedFileAndPosition() with unknown ID"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        _addEntryToFile([1,1,5,1,10], f)
        _addEntryToFile([2,1,15,2,20], f)
        _addEntryToFile([3,2,25,2,30], f)
        _addEntryToFile([4,0,-1,0,-1], f)

    # When
    provider = IndexProvider(filename)

    # Then
    with pytest.raises(IdMismatchException):
        provider.setSedFileAndPosition(5, 2, 30)

###############################################################################

def test_setSedFileAndPosition_alreadySet(temp_dir_fixture):
    """Tests the setSedFileAndPosition() for the case the position is already set"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        _addEntryToFile([1,1,5,1,10], f)
        _addEntryToFile([2,1,15,2,20], f)
        _addEntryToFile([3,2,25,2,30], f)
        _addEntryToFile([4,0,-1,0,-1], f)

    # When
    provider = IndexProvider(filename)

    # Then
    with pytest.raises(AlreadySetException):
        provider.setSedFileAndPosition(3, 2, 30)

###############################################################################

def test_setSedPosition_success(temp_dir_fixture):
    """Test successful call of the setSedPosition()"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        _addEntryToFile([1,1,5,1,10], f)
        _addEntryToFile([2,1,15,2,20], f)
        _addEntryToFile([3,2,25,2,30], f)
        _addEntryToFile([4,0,-1,0,-1], f)

    # When
    expected_file = 2
    expected_pos = 35
    provider = IndexProvider(filename)
    provider.setSedFileAndPosition(4, expected_file, expected_pos)
    info = provider.getFilesAndPositions(4)

    # Then
    assert info[0] == expected_file
    assert info[1] == expected_pos
    assert info[2] == 0
    assert info[3] == -1
    with open(filename, 'rb') as f:
        f.seek(3 * 28)
        assert np.fromfile(f, count=1, dtype=np.int64)[0] == 4
        assert np.fromfile(f, count=1, dtype=np.uint16)[0] == expected_file
        assert np.fromfile(f, count=1, dtype=np.int64)[0] == expected_pos
        assert np.fromfile(f, count=1, dtype=np.uint16)[0] == 0
        assert np.fromfile(f, count=1, dtype=np.int64)[0] == -1

###############################################################################

def test_setPdzFileAndPosition_idMismatch(temp_dir_fixture):
    """Tests the setPdzFileAndPosition() with unknown ID"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        _addEntryToFile([1,1,5,1,10], f)
        _addEntryToFile([2,1,15,2,20], f)
        _addEntryToFile([3,2,25,2,30], f)
        _addEntryToFile([4,0,-1,0,-1], f)

    # When
    provider = IndexProvider(filename)

    # Then
    with pytest.raises(IdMismatchException):
        provider.setPdzFileAndPosition(5, 2, 40)

###############################################################################

def test_setPdzFileAndPosition_alreadySet(temp_dir_fixture):
    """Tests the setPdzFileAndPosition() for the case the position is already set"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        _addEntryToFile([1,1,5,1,10], f)
        _addEntryToFile([2,1,15,2,20], f)
        _addEntryToFile([3,2,25,2,30], f)
        _addEntryToFile([4,0,-1,0,-1], f)

    # When
    provider = IndexProvider(filename)

    # Then
    with pytest.raises(AlreadySetException):
        provider.setPdzFileAndPosition(3, 2, 40)

###############################################################################

def test_setPdzFileAndPosition_success(temp_dir_fixture):
    """Test successful call of the setPdzFileAndPosition()"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        _addEntryToFile([1,1,5,1,10], f)
        _addEntryToFile([2,1,15,2,20], f)
        _addEntryToFile([3,2,25,2,30], f)
        _addEntryToFile([4,0,-1,0,-1], f)
    expected_file = 2
    expected_pos = 40

    # When
    provider = IndexProvider(filename)
    provider.setPdzFileAndPosition(4, expected_file, expected_pos)
    info = provider.getFilesAndPositions(4)

    # Then
    assert info[0] == 0
    assert info[1] == -1
    assert info[2] == expected_file
    assert info[3] == expected_pos
    with open(filename, 'rb') as f:
        f.seek(3 * 28)
        assert np.fromfile(f, count=1, dtype=np.int64)[0] == 4
        assert np.fromfile(f, count=1, dtype=np.uint16)[0] == 0
        assert np.fromfile(f, count=1, dtype=np.int64)[0] == -1
        assert np.fromfile(f, count=1, dtype=np.uint16)[0] == expected_file
        assert np.fromfile(f, count=1, dtype=np.int64)[0] == expected_pos

###############################################################################

def test_size(temp_dir_fixture):
    """Test calling the size() method"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        _addEntryToFile([1,1,5,1,10], f)
        _addEntryToFile([2,1,15,2,20], f)
        _addEntryToFile([3,2,25,2,30], f)
        _addEntryToFile([4,0,-1,0,-1], f)

    # When
    provider = IndexProvider(filename)
    s = provider.size()

    # Then
    assert s == 4

###############################################################################

def test_missingSedList(temp_dir_fixture):
    """Test the missingSedList() method"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        _addEntryToFile([1,1,5,1,10], f)
        _addEntryToFile([2,1,15,2,20], f)
        _addEntryToFile([3,0,-1,1,30], f)
        _addEntryToFile([4,2,25,2,40], f)
        _addEntryToFile([5,0,-1,1,50], f)

    # When
    provider = IndexProvider(filename)
    missing = provider.missingSedList()

    # Then
    assert len(missing) == 2
    assert missing.index(3) != -1
    assert missing.index(5) != -1

###############################################################################

def test_missingPdzList(temp_dir_fixture):
    """Test the missingSedList() method"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.bin')
    with open(filename, 'wb') as f:
        _addEntryToFile([1,1,5,1,10], f)
        _addEntryToFile([2,1,15,2,20], f)
        _addEntryToFile([3,2,25,0,-1], f)
        _addEntryToFile([4,2,35,2,30], f)
        _addEntryToFile([5,2,45,0,-1], f)

    # When
    provider = IndexProvider(filename)
    missing = provider.missingPdzList()

    # Then
    assert len(missing) == 2
    assert missing.index(3) != -1
    assert missing.index(5) != -1

###############################################################################

