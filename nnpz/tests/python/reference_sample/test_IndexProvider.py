#
# Copyright (C) 2012-2022 Euclid Science Ground Segment
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


import os

import numpy as np
import pytest
from nnpz.exceptions import *
from nnpz.reference_sample.IndexProvider import IndexProvider

# noinspection PyUnresolvedReferences
from .fixtures import temp_dir_fixture


@pytest.fixture
def index_data():
    return np.array(
        [(1, 0, 1), (2, 1, 2), (3, 0, 3), (4, 2, 4)],
        dtype=[('id', np.int64), ('sed_file', np.int64), ('sed_offset', np.int64)]
    )


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

def test_constructor_idMismatch(temp_dir_fixture, index_data):
    """
    Tests the case of a duplicate ID
    """

    # Given
    filename = os.path.join(temp_dir_fixture, 'duplicate.npy')
    index_data['id'][-1] = index_data['id'][0]
    np.save(filename, index_data)

    # Then
    with pytest.raises(DuplicateIdException):
        IndexProvider(filename)


###############################################################################

def test_getIdList(temp_dir_fixture, index_data):
    """Tests the getIds() returns the correct IDs"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.npy')
    np.save(filename, index_data)

    # When
    provider = IndexProvider(filename)
    id_list = provider.get_ids()

    # Then
    np.testing.assert_array_equal(id_list, [1, 2, 3, 4])


###############################################################################

def test_getFileList(temp_dir_fixture, index_data):
    """
    Tests the getFiles() returns the correct info
    """

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.npy')
    np.save(filename, index_data)

    # When
    provider = IndexProvider(filename)
    file_list = sorted(list(provider.get_files('sed')))

    # Then
    np.testing.assert_array_equal(file_list, [0, 1, 2])


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
        info = provider.get(row[0], 'sed')
        # Then
        assert info.file == row[1]
        assert info.offset == row[2]
        # When
        info = provider.get(row[0], 'pdz')
        # Then
        assert info is None


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
    assert provider.get(5, 'sed') is None


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
        provider.add(3, 'sed', IndexProvider.ObjectLocation(2, 40))


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
    provider.add(5, 'sed', IndexProvider.ObjectLocation(expected_file, expected_pos))
    provider.add(5, 'pdz', IndexProvider.ObjectLocation(expected_file + 1, expected_pos * 2))
    info_sed = provider.get(5, 'sed')
    info_pdz = provider.get(5, 'pdz')

    # Then
    assert info_sed.file == expected_file
    assert info_sed.offset == expected_pos
    assert info_pdz.file == expected_file + 1
    assert info_pdz.offset == expected_pos * 2
    for row in index_data:
        info = provider.get(row[0], 'sed')
        assert info.file == row[1]
        assert info.offset == row[2]
        info = provider.get(row[0], 'pdz')
        assert info is None


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
        provider.add(5, 'sed', IndexProvider.ObjectLocation(expected_file, expected_pos))

    provider = IndexProvider(filename)
    info = provider.get(5, 'sed')

    # Then
    assert info.file == expected_file
    assert info.offset == expected_pos
    for row in index_data:
        info = provider.get(row[0], 'sed')
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


###############################################################################

def test_bulkAppend(temp_dir_fixture, index_data):
    """
    Test doing a bulk import
    """

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.npy')
    np.save(filename, index_data)
    expected = np.array([(10, 4, 0), (11, 4, 1), (12, 5, 3), (13, 5, 4)], dtype=index_data.dtype)

    # When
    with IndexProvider(filename) as provider:
        provider.bulk_add(expected)

    # Then
    provider = IndexProvider(filename)
    assert len(provider) == 8


###############################################################################

def test_bulkAppendDuplicate(temp_dir_fixture, index_data):
    """
    Test doing a bulk import with a duplicate ID
    """

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.npy')
    np.save(filename, index_data)
    new_data = np.array([(10, 4, 0), (11, 4, 1), (2, 5, 3), (3, 5, 4)], dtype=index_data.dtype)

    # When
    provider = IndexProvider(filename)

    # Then
    with pytest.raises(DuplicateIdException):
        provider.bulk_add(new_data)


###############################################################################

def test_bulkAppendVertical(temp_dir_fixture, index_data):
    """
    Bulk-add another index with duplicated IDs but that index a different data set
    """

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.npy')
    np.save(filename, index_data)
    new_data = np.array(
        [(1, 10, 0), (2, 11, 1), (3, 10, 5), (4, 12, 8)],
        dtype=[('id', np.int64), ('pdz_file', np.int64), ('pdz_offset', np.int64)]
    )

    # When
    provider = IndexProvider(filename)
    provider.bulk_add(new_data)

    # Then
    np.testing.assert_array_equal(provider.get_ids(), [1, 2, 3, 4])


###############################################################################

def test_bulkAppendMixed(temp_dir_fixture, index_data):
    """
    Bulk-add another index combining ids that are duplicated an others not
    """

    # Given
    filename = os.path.join(temp_dir_fixture, 'index.npy')
    np.save(filename, index_data)
    new_data = np.array(
        [(1, 10, 0), (2, 11, 1), (10, 10, 5), (11, 12, 8)],
        dtype=[('id', np.int64), ('pdz_file', np.int64), ('pdz_offset', np.int64)]
    )

    # When
    provider = IndexProvider(filename)
    provider.bulk_add(new_data)

    # Then
    np.testing.assert_array_equal(provider.get_ids(), [1, 2, 3, 4, 10, 11])
