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
from nnpz.reference_sample.PdzDataProvider import PdzDataProvider

from .fixtures import *

###############################################################################

def test_constructor_missingFile(temp_dir_fixture):
    """Tests the case that the file does not exist"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'missing')

    # Then
    with pytest.raises(FileNotFoundException):
        PdzDataProvider(filename)

###############################################################################

def test_setRedshiftBins_notSetBefore(temp_dir_fixture):
    """Tests that if the file has no header the setRedshftBins() populates it correctly"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'empty.bin')
    open(filename, 'w').close()
    expected_bins = np.asarray([1,2,3,4], dtype=np.float32)

    # When
    provider = PdzDataProvider(filename)
    provider.setRedshiftBins(expected_bins)

    # Then
    assert os.path.getsize(filename) == 4 + 4 * len(expected_bins)
    with open(filename, 'rb') as f:
        length = np.fromfile(f, count=1, dtype=np.uint32)[0]
        assert length == len(expected_bins)
        file_bins = np.fromfile(f, count=length, dtype=np.float32)
        assert np.array_equal(file_bins, expected_bins)

###############################################################################

def test_setRedshiftBins_alreadySet(pdz_data_files_fixture):
    """Tests that if the bins are already set we get exception"""

    # Given
    bins = np.asarray([1,2,3,4], dtype=np.float32)

    # When
    provider = PdzDataProvider(pdz_data_files_fixture[1])

    # Then
    with pytest.raises(AlreadySetException):
        provider.setRedshiftBins(bins)

###############################################################################

def test_setRedshiftBins_invalidDimension(temp_dir_fixture):
    """Tests that we get exception for wrong dimesionality"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'empty.bin')
    open(filename, 'w').close()
    bins = np.zeros((2,4), dtype=np.float32)

    # When
    provider = PdzDataProvider(filename)

    # Then
    with pytest.raises(InvalidDimensionsException):
        provider.setRedshiftBins(bins)

###############################################################################

def test_setRedshiftBins_nonIncreasingValues(temp_dir_fixture):
    """Tests that we get exception for non increasing bins"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'empty.bin')
    open(filename, 'w').close()
    bins = np.asarray([1,2,4,3,5], dtype=np.float32)

    # When
    provider = PdzDataProvider(filename)

    # Then
    with pytest.raises(InvalidAxisException):
        provider.setRedshiftBins(bins)

###############################################################################

def test_getRedshiftBins_alreadySet(pdz_data_files_fixture, redshift_bins_fixture):
    """Tests that if the bins are already set we get the correct values"""

    # Given
    provider = PdzDataProvider(pdz_data_files_fixture[1])

    # When
    bins = provider.getRedshiftBins()

    # Then
    assert np.array_equal(bins, redshift_bins_fixture)

###############################################################################

def test_getRedshiftBins_noHeader(temp_dir_fixture):
    """Tests that if the bins are already set we get the correct values"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'empty.bin')
    open(filename, 'w').close()
    provider = PdzDataProvider(filename)

    # When
    bins = provider.getRedshiftBins()

    # Then
    assert bins is None

###############################################################################

def test_readPdz_success(pdz_data_files_fixture, pdz_list_fixture, redshift_bins_fixture):
    """Tests successful call of readPdz()"""

    # Given
    provider = PdzDataProvider(pdz_data_files_fixture[1])

    pos = 4 + 4 * len(redshift_bins_fixture)
    for expected_id, expected_data in pdz_list_fixture[1]:

        # When
        found_id, found_data = provider.readPdz(pos)

        # Then
        assert found_id == expected_id
        assert np.array_equal(found_data, expected_data)

        pos += 8 + 4 * len(redshift_bins_fixture)

###############################################################################

def test_readPdz_uninitialized(temp_dir_fixture):
    """Test that calling readPdz() to a file without header raises an exception"""

    #Given
    filename = os.path.join(temp_dir_fixture, 'empty.bin')
    open(filename, 'w').close()

    # When
    provider = PdzDataProvider(filename)

    # Then
    with pytest.raises(UninitializedException):
        provider.readPdz(0)

###############################################################################

def test_appendPdz_success(pdz_data_files_fixture, redshift_bins_fixture):
    """Test that successful call of appendPdz() adds the PDZ in the file"""

    #Given
    expected_pos = os.path.getsize(pdz_data_files_fixture[1])
    expected_id = 56
    expected_data = np.asarray(range(len(redshift_bins_fixture)), dtype=np.float32)

    # When
    provider = PdzDataProvider(pdz_data_files_fixture[1])
    pos = provider.appendPdz(expected_id, expected_data)

    # Then
    assert pos == expected_pos
    with open(pdz_data_files_fixture[1], 'rb') as f:
        f.seek(pos)
        found_id = np.fromfile(f, count=1, dtype=np.int64)[0]
        assert found_id == expected_id
        found_data = np.fromfile(f, count=len(redshift_bins_fixture), dtype=np.float32)
        assert np.array_equal(found_data, expected_data)

###############################################################################

def test_appendPdz_uninitialized(temp_dir_fixture, redshift_bins_fixture):
    """Test that calling appendPdz() to a file without header raises an exception"""

    #Given
    filename = os.path.join(temp_dir_fixture, 'empty.bin')
    open(filename, 'w').close()
    expected_id = 56
    expected_data = np.asarray(range(len(redshift_bins_fixture)), dtype=np.float32)

    # When
    provider = PdzDataProvider(filename)

    # Then
    with pytest.raises(UninitializedException):
        provider.appendPdz(expected_id, expected_data)

###############################################################################

def test_appendPdz_wrongLength(pdz_data_files_fixture, redshift_bins_fixture):
    """Test that appendPdz() raises an exception for wrong data length"""

    #Given
    expected_pos = os.path.getsize(pdz_data_files_fixture[1])
    expected_id = 56
    expected_data = np.asarray(range(len(redshift_bins_fixture) - 1), dtype=np.float32)

    # When
    provider = PdzDataProvider(pdz_data_files_fixture[1])

    # Then
    with pytest.raises(InvalidDimensionsException):
        provider.appendPdz(expected_id, expected_data)

###############################################################################

def test_appendPdz_wrongDimensionality(pdz_data_files_fixture, redshift_bins_fixture):
    """Test that appendPdz() raises an exception for wrong data dimensionality"""

    #Given
    expected_pos = os.path.getsize(pdz_data_files_fixture[1])
    expected_id = 56
    expected_data = np.zeros((len(redshift_bins_fixture), 2), dtype=np.float32)

    # When
    provider = PdzDataProvider(pdz_data_files_fixture[1])

    # Then
    with pytest.raises(InvalidDimensionsException):
        provider.appendPdz(expected_id, expected_data)

###############################################################################

def test_validate_diffInputSize(pdz_data_files_fixture):
    """Test that the validate() raises an exception if the id and pos lists have different size"""

    # Given
    id_list = [1, 2, 3]
    pos_list = [1, 2, 3, 4]

    # When
    provider = PdzDataProvider(pdz_data_files_fixture[1])

    # Then
    with pytest.raises(InvalidDimensionsException):
        provider.validate(id_list, pos_list)

###############################################################################

def test_validate_consistentFile(pdz_data_files_fixture, pdz_list_fixture):
    """Test that when the file is consistent with the given lists it returns None"""

    # Given
    id_list = [x for x,_ in pdz_list_fixture[1]]
    pos_offset = 4 + 4 * len(pdz_list_fixture[1][0][1])
    pos_shift = 8 + 4 * len(pdz_list_fixture[1][0][1])
    pos_list = [pos_offset + i * pos_shift for i in range(len(pdz_list_fixture[1]))]

    # When
    provider = PdzDataProvider(pdz_data_files_fixture[1])
    message = provider.validate(id_list, pos_list)

    # Then
    assert message is None

###############################################################################

def test_validate_posOutOfFile(pdz_data_files_fixture, pdz_list_fixture):
    """Test the case where a position is out of the file"""

    # Given
    id_list = [x for x,_ in pdz_list_fixture[1]]
    pos_offset = 4 + 4 * len(pdz_list_fixture[1][0][1])
    pos_shift = 8 + 4 * len(pdz_list_fixture[1][0][1])
    pos_list = [pos_offset + i * pos_shift for i in range(len(pdz_list_fixture[1]))]
    file_size = os.path.getsize(pdz_data_files_fixture[1])
    pos_list[1] = file_size

    # When
    provider = PdzDataProvider(pdz_data_files_fixture[1])
    message = provider.validate(id_list, pos_list)

    # Then
    assert message == 'Position ({}) out of file for ID={}'.format(file_size, id_list[1])

###############################################################################

def test_validate_incosistentId(pdz_data_files_fixture, pdz_list_fixture):
    """Test the case where an ID differs"""

    # Given
    id_list = [x for x,_ in pdz_list_fixture[1]]
    pos_offset = 4 + 4 * len(pdz_list_fixture[1][0][1])
    pos_shift = 8 + 4 * len(pdz_list_fixture[1][0][1])
    pos_list = [pos_offset + i * pos_shift for i in range(len(pdz_list_fixture[1]))]
    old_id = id_list[1]
    id_list[1] += 1

    # When
    provider = PdzDataProvider(pdz_data_files_fixture[1])
    message = provider.validate(id_list, pos_list)

    # Then
    assert message == 'Inconsistent IDs (' + str(id_list[1]) + ', ' + str(old_id) + ')'

###############################################################################

def test_validate_exceedFile(pdz_data_files_fixture, pdz_list_fixture):
    """Test the case where a PDZ goes outside of the file"""

    # Given
    id_list = [x for x,_ in pdz_list_fixture[1]]
    pos_offset = 4 + 4 * len(pdz_list_fixture[1][0][1])
    pos_shift = 8 + 4 * len(pdz_list_fixture[1][0][1])
    pos_list = [pos_offset + i * pos_shift for i in range(len(pdz_list_fixture[1]))]
    extra_id = 88
    id_list.append(extra_id)
    pos_list.append(pos_list[-1] + pos_shift)
    with open(pdz_data_files_fixture[1], 'ab') as f:
        np.asarray([extra_id], dtype=np.int64).tofile(f)

    # When
    provider = PdzDataProvider(pdz_data_files_fixture[1])
    message = provider.validate(id_list, pos_list)

    # Then
    assert message == 'Data length bigger than file for ID={}'.format(extra_id)

###############################################################################

def test_validate_uninitialized(temp_dir_fixture, pdz_list_fixture):
    """Test the case where the last PDZ is ouside of the file"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'empty.bin')
    open(filename, 'w').close()
    id_list = [x for x,_ in pdz_list_fixture[1]]
    pos_offset = 4 + 4 * len(pdz_list_fixture[1][0][1])
    pos_shift = 8 + 4 * len(pdz_list_fixture[1][0][1])
    pos_list = [pos_offset + i * pos_shift for i in range(len(pdz_list_fixture[1]))]

    # When
    provider = PdzDataProvider(filename)

    # Then
    with pytest.raises(UninitializedException):
        message = provider.validate(id_list, pos_list)
