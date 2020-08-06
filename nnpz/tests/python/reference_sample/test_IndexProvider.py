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

import os

import numpy as np
import pytest
from nnpz.exceptions import *
from nnpz.reference_sample.IndexProvider import IndexProvider

# noinspection PyUnresolvedReferences
from .fixtures import temp_dir_fixture


@pytest.fixture
def index_data():
    return np.array([[1, 0, 1], [2, 1, 2], [3, 0, 3], [4, 2, 4]])


###############################################################################

def test_constructor_malformed(temp_dir_fixture):
    """
    Tests the case that the index is malformed
    """

    # Given
    filename = os.path.join(temp_dir_fixture, 'malformed.npy')
    with open(filename, 'wb') as f:
        np.asarray([1, 2, 3, 4], dtype=np.int64).tofile(f)

    # Then
    with pytest.raises(CorruptedFileException):
        IndexProvider(filename)


###############################################################################

def test_constructor_bad_shape(temp_dir_fixture):
    """
    The index has the wrong shape
    """

    # Given
    filename_01 = os.path.join(temp_dir_fixture, 'bad_shape_01.npy')
    filename_02 = os.path.join(temp_dir_fixture, 'bad_shape_02.npy')
    np.save(filename_01, np.arange(10))
    np.save(filename_02, np.arange(10).reshape(5, 2))

    # When
    with pytest.raises(CorruptedFileException):
        IndexProvider(filename_01)

    with pytest.raises(CorruptedFileException):
        IndexProvider(filename_02)


###############################################################################

def test_constructor_idMismatch(temp_dir_fixture):
    """
    Tests the case of a duplicate ID
    """

    # Given
    filename = os.path.join(temp_dir_fixture, 'duplicate.npy')
    np.save(filename, np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3], [2, 0, 4]]))

    # Then
    with pytest.raises(DuplicateIdException):
        IndexProvider(filename)


###############################################################################

def test_getIdList(temp_dir_fixture):
    """Tests the getIds() returns the correct IDs"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.npy')
    np.save(filename, np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 0, 4]]))

    # When
    provider = IndexProvider(filename)
    id_list = provider.getIds()

    # Then
    assert np.array_equal(id_list, [1, 2, 3, 4])


###############################################################################

def test_getFileList(temp_dir_fixture):
    """
    Tests the getFiles() returns the correct info
    """

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.npy')
    np.save(filename, np.array([[1, 5, 1], [2, 10, 2], [3, 15, 3], [4, 20, 4]]))

    # When
    provider = IndexProvider(filename)
    file_list = sorted(list(provider.getFiles()))

    # Then
    assert np.array_equal(file_list, [5, 10, 15, 20])


###############################################################################

def test_get_success(temp_dir_fixture, index_data):
    """
    Tests the getFilesAndPositions() returns the correct info for all IDs
    """

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.npy')
    np.save(filename, index_data)

    provider = IndexProvider(filename)

    for row in index_data:
        # When
        info = provider.get(row[0])
        # Then
        assert info.file == row[1]
        assert info.offset == row[2]


###############################################################################

def test_get_idMismatch(temp_dir_fixture, index_data):
    """
    Tests the get() with unknown ID
    """

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.npy')
    np.save(filename, index_data)

    # When
    provider = IndexProvider(filename)

    # Then
    assert provider.get(5) is None


###############################################################################

def test_add_alreadySet(temp_dir_fixture, index_data):
    """
    Tests the add() for the case the position is already set
    """

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.npy')
    np.save(filename, index_data)

    # When
    provider = IndexProvider(filename)

    # Then
    with pytest.raises(DuplicateIdException):
        provider.add(3, IndexProvider.ObjectLocation(2, 40))


###############################################################################

def test_add_success(temp_dir_fixture, index_data):
    """
    Test successful call of the add() method
    """

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.npy')
    np.save(filename, index_data)
    expected_file = 2
    expected_pos = 40

    # When
    provider = IndexProvider(filename)
    provider.add(5, IndexProvider.ObjectLocation(expected_file, expected_pos))
    info = provider.get(5)

    # Then
    assert info.file == expected_file
    assert info.offset == expected_pos
    for row in index_data:
        info = provider.get(row[0])
        assert info.file == row[1]
        assert info.offset == row[2]


###############################################################################

def test_add_success_with(temp_dir_fixture, index_data):
    """
    Test successful call of the add() method
    """

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.npy')
    np.save(filename, index_data)
    expected_file = 2
    expected_pos = 40

    # When
    with IndexProvider(filename) as provider:
        provider.add(5, IndexProvider.ObjectLocation(expected_file, expected_pos))

    provider = IndexProvider(filename)
    info = provider.get(5)

    # Then
    assert info.file == expected_file
    assert info.offset == expected_pos
    for row in index_data:
        info = provider.get(row[0])
        assert info.file == row[1]
        assert info.offset == row[2]


###############################################################################

def test_size(temp_dir_fixture, index_data):
    """
    Test calling the __len__() method"
    """

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.npy')
    np.save(filename, index_data)

    # When
    provider = IndexProvider(filename)
    s = len(provider)

    # Then
    assert s == 4
