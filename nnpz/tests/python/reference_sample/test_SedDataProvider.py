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

from nnpz.exceptions import *
from nnpz.reference_sample.SedDataProvider import SedDataProvider

from .fixtures import *


###############################################################################

def test_corrupted(temp_dir_fixture):
    """
    Test reading a corrupted SED file
    """

    # Given
    filename = os.path.join(temp_dir_fixture, 'sed_data_1.npy')
    with open(filename, 'wt') as fd:
        fd.write('1234')

    # Then
    with pytest.raises(CorruptedFileException):
        SedDataProvider(filename)


###############################################################################

def test_bad_shape(temp_dir_fixture):
    """
    Test reading a SED file with the wrong shape
    """
    # Given
    filename_01 = os.path.join(temp_dir_fixture, 'bad_shape_01.npy')
    filename_02 = os.path.join(temp_dir_fixture, 'bad_shape_02.npy')
    np.save(filename_01, np.arange(10))
    np.save(filename_02, np.arange(60).reshape(5, 3, 4))

    # When
    with pytest.raises(CorruptedFileException):
        SedDataProvider(filename_01)

    with pytest.raises(CorruptedFileException):
        SedDataProvider(filename_02)


###############################################################################

def test_readSed_success(sed_data_files_fixture, sed_list_fixture):
    """
    Tests successful call of readSed()
    """

    # Given
    provider = SedDataProvider(sed_data_files_fixture[1])

    pos = 0
    for _, expected_data in sed_list_fixture[1]:
        # When
        found_data = provider.read_sed(pos)

        # Then
        np.testing.assert_array_equal(found_data, expected_data)

        pos += 1


###############################################################################

def test_appendSed_success(sed_data_files_fixture):
    """
    Tests successful call of appendSed()
    """

    # Given
    expected_data = np.asarray([(1., 10,), (2., 20.)], dtype=np.float32)

    # When
    with SedDataProvider(sed_data_files_fixture[1]) as provider:
        pos = provider.append_sed(expected_data)

    # Then
    with SedDataProvider(sed_data_files_fixture[1]) as provider:
        data = provider.read_sed(pos)
        np.testing.assert_array_equal(expected_data, data)


###############################################################################

def test_appendSed_invalidDataDimension(sed_data_files_fixture):
    """
    Test that the appendSed() method raises an exception for wrong data dimensions
    """

    # Given
    four_dim_data = np.zeros((100, 2, 2, 8), dtype=np.float32)
    wrong_second_dimension = np.zeros((100, 3), dtype=np.float32)

    # When
    provider = SedDataProvider(sed_data_files_fixture[1])

    # Then
    with pytest.raises(InvalidDimensionsException):
        provider.append_sed(four_dim_data)
    with pytest.raises(InvalidDimensionsException):
        provider.append_sed(wrong_second_dimension)


###############################################################################

def test_appendSed_nonIncreasingWavelength(sed_data_files_fixture):
    """
    Test that the appendSed() method raises an exception for non increasing wavelengths
    """

    # Given
    wrong_wavelength = np.asarray([(1., 10,), (3., 20.), (2., 30)], dtype=np.float32)

    # When
    provider = SedDataProvider(sed_data_files_fixture[1])

    # Then
    with pytest.raises(InvalidAxisException):
        provider.append_sed(wrong_wavelength)


###############################################################################


def test_appendBulkSed(sed_data_files_fixture):
    """
    Test that appendSed() can append multiple sed at once
    """

    # Given
    new_sed = np.asarray([
        [(1., 100.), (1.5, 168.3), ],
        [(1., 400.), (1.5, 700.), ],
    ])

    # When
    provider = SedDataProvider(sed_data_files_fixture[1])

    # Then
    offsets = provider.append_sed(new_sed)
    assert len(offsets) == len(new_sed)
    for i, o in enumerate(offsets):
        sed = provider.read_sed(o)
        np.testing.assert_allclose(new_sed[i], sed)
