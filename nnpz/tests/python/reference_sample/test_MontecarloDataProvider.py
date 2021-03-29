#
# Copyright (C) 2012-2021 Euclid Science Ground Segment
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

from __future__ import division, print_function

from nnpz.exceptions import *
from nnpz.reference_sample.MontecarloDataProvider import MontecarloDataProvider

from .fixtures import *


###############################################################################

def test_corrupted(temp_dir_fixture):
    """
    Test reading a corrupted MC file
    """

    # Given
    filename = os.path.join(temp_dir_fixture, 'mc_data_1.npy')
    with open(filename, 'wt') as fd:
        fd.write('1234')

    # Then
    with pytest.raises(CorruptedFileException):
        MontecarloDataProvider(filename)


###############################################################################

def test_readMc_success(mc_data_files_fixture, mc_data_fixture):
    """
    Tests successful call of read()
    """

    # Given
    provider = MontecarloDataProvider(mc_data_files_fixture[1])

    for i, expected_data in enumerate(mc_data_fixture[1]):
        # When
        found_data = provider.read(i)

        # Then
        assert found_data.shape == (100,)


###############################################################################

def test_readMc_uninitialized(temp_dir_fixture):
    """
    Test that calling read() to a file without data raises an exception
    """

    # Given
    filename = os.path.join(temp_dir_fixture, 'empty.npy')

    # When
    provider = MontecarloDataProvider(filename)

    # Then
    with pytest.raises(UninitializedException):
        provider.read(0)


###############################################################################

def test_appendMc_empty_success(temp_dir_fixture):
    """
    Test that successful call of append() adds the sample in the file
    """

    # Given
    filename = os.path.join(temp_dir_fixture, 'empty.npy')
    expected_data = np.random.rand(100, 4)

    # When
    with MontecarloDataProvider(filename) as provider:
        pos = provider.append(expected_data)

    # Then
    array = np.load(filename)
    assert np.array_equal(array[pos].reshape(100, 4), expected_data)


###############################################################################

def test_appendMc_success(mc_data_files_fixture):
    """
    Test that successful call of append() adds the sample in the file
    """

    # Given
    expected_data = np.zeros(100, dtype=[
        ('A', np.float32), ('B', np.float32), ('C', np.float32), ('D', np.float32)
    ])
    for c in expected_data.dtype.names:
        expected_data[c] = np.random.rand(*expected_data.shape)

    # When
    with MontecarloDataProvider(mc_data_files_fixture[1]) as provider:
        pos = provider.append(expected_data)

    # Then
    array = np.load(mc_data_files_fixture[1])
    assert np.array_equal(array[pos].reshape(100,), expected_data)


###############################################################################

def test_append_wrongDimensionality(mc_data_files_fixture):
    """
    Test that append() raises an exception for wrong data dimensions
    """

    # Given
    bad_n_samples = np.random.rand(50, 4)
    bad_dimensions = np.random.rand(100, 5)

    # When
    provider = MontecarloDataProvider(mc_data_files_fixture[1])

    # Then
    with pytest.raises(InvalidDimensionsException):
        provider.append(bad_n_samples)
    with pytest.raises(InvalidDimensionsException):
        provider.append(bad_dimensions)


###############################################################################

def test_appendPdz_wrongShape(mc_data_files_fixture):
    """
    Test that append() raises an exception for wrong data shape
    """

    # Given
    bad_data = np.random.rand(100, 5, 4)

    # When
    provider = MontecarloDataProvider(mc_data_files_fixture[1])

    # Then
    with pytest.raises(InvalidDimensionsException):
        provider.append(bad_data)


###############################################################################

def test_appendBulk(mc_data_files_fixture):
    """
    Test that append() can append multiple objects at once
    """

    # Given
    expected_data = np.zeros((20, 100), dtype=[
        ('A', np.float32), ('B', np.float32), ('C', np.float32), ('D', np.float32)
    ])
    for c in expected_data.dtype.names:
        expected_data[c] = np.random.rand(*expected_data.shape)

    # When
    provider = MontecarloDataProvider(mc_data_files_fixture[1])

    # Then
    offsets = provider.append(expected_data)
    assert len(offsets) == len(expected_data)
    for i, o in enumerate(offsets):
        data = provider.read(o)
        assert all([np.allclose(expected_data[c][i], data[c]) for c in data.dtype.names])
