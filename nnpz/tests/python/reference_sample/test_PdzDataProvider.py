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

"""
Created on: 10/11/17
Author: Nikolaos Apostolakos
"""


from nnpz.exceptions import *
from nnpz.reference_sample.PdzDataProvider import PdzDataProvider

from .fixtures import *


###############################################################################

def test_corrupted(temp_dir_fixture):
    """
    Test reading a corrupted SED file
    """

    # Given
    filename = os.path.join(temp_dir_fixture, 'pdz_data_1.npy')
    with open(filename, 'wt') as fd:
        fd.write('1234')

    # Then
    with pytest.raises(CorruptedFileException):
        PdzDataProvider(filename)


###############################################################################

def test_bad_shape(temp_dir_fixture):
    """
    Test reading a SED file with the wrong shape
    """
    # Given
    filename_01 = os.path.join(temp_dir_fixture, 'bad_shape_01.npy')
    filename_02 = os.path.join(temp_dir_fixture, 'bad_shape_02.npy')
    np.save(filename_01, np.arange(10))
    np.save(filename_02, np.arange(30).reshape(5, 3, 2))

    # When
    with pytest.raises(CorruptedFileException):
        PdzDataProvider(filename_01)

    with pytest.raises(CorruptedFileException):
        PdzDataProvider(filename_02)


###############################################################################

def test_setRedshiftBins_notSetBefore(temp_dir_fixture):
    """
    Tests that if the file has no header the setRedshftBins() populates it correctly
    """

    # Given
    filename = os.path.join(temp_dir_fixture, 'empty.npy')
    expected_bins = np.asarray([1, 2, 3, 4], dtype=np.float32)

    # When
    with PdzDataProvider(filename) as provider:
        provider.setRedshiftBins(expected_bins)

    # Then
    array = np.load(filename)
    assert array.shape[1] == len(expected_bins)
    assert np.array_equal(array[0, :], expected_bins)


###############################################################################

def test_setRedshiftBins_alreadySet(pdz_data_files_fixture):
    """
    Tests that if the bins are already set we get exception
    """

    # Given
    bins = np.asarray([1, 2, 3, 4], dtype=np.float32)

    # When
    provider = PdzDataProvider(pdz_data_files_fixture[1])

    # Then
    with pytest.raises(AlreadySetException):
        provider.setRedshiftBins(bins)


###############################################################################

def test_setRedshiftBins_invalidDimension(temp_dir_fixture):
    """
    Tests that we get exception for wrong dimensionality
    """

    # Given
    filename = os.path.join(temp_dir_fixture, 'empty.npy')
    bins = np.zeros((2, 4), dtype=np.float32)

    # When
    provider = PdzDataProvider(filename)

    # Then
    with pytest.raises(InvalidDimensionsException):
        provider.setRedshiftBins(bins)


###############################################################################

def test_setRedshiftBins_nonIncreasingValues(temp_dir_fixture):
    """
    Tests that we get exception for non increasing bins
    """

    # Given
    filename = os.path.join(temp_dir_fixture, 'empty.npy')
    bins = np.asarray([1, 2, 4, 3, 5], dtype=np.float32)

    # When
    provider = PdzDataProvider(filename)

    # Then
    with pytest.raises(InvalidAxisException):
        provider.setRedshiftBins(bins)


###############################################################################

def test_getRedshiftBins_alreadySet(pdz_data_files_fixture, redshift_bins_fixture):
    """
    Tests that if the bins are already set we get the correct values
    """

    # Given
    provider = PdzDataProvider(pdz_data_files_fixture[1])

    # When
    bins = provider.getRedshiftBins()

    # Then
    assert np.array_equal(bins, redshift_bins_fixture)


###############################################################################

def test_getRedshiftBins_noHeader(temp_dir_fixture):
    """
    Get the bins on an empty file
    """

    # Given
    filename = os.path.join(temp_dir_fixture, 'empty.npy')
    provider = PdzDataProvider(filename)

    # When
    bins = provider.getRedshiftBins()

    # Then
    assert bins is None


###############################################################################

def test_readPdz_success(pdz_data_files_fixture, pdz_list_fixture, redshift_bins_fixture):
    """
    Tests successful call of readPdz()
    """

    # Given
    provider = PdzDataProvider(pdz_data_files_fixture[1])

    for i, expected_data in enumerate(pdz_list_fixture[1]):
        # When
        found_data = provider.readPdz(i + 1)

        # Then
        assert np.array_equal(found_data, expected_data[1])


###############################################################################

def test_readPdz_uninitialized(temp_dir_fixture):
    """
    Test that calling readPdz() to a file without header raises an exception
    """

    # Given
    filename = os.path.join(temp_dir_fixture, 'empty.npy')

    # When
    provider = PdzDataProvider(filename)

    # Then
    with pytest.raises(UninitializedException):
        provider.readPdz(0)


###############################################################################

def test_appendPdz_success(pdz_data_files_fixture, redshift_bins_fixture):
    """
    Test that successful call of appendPdz() adds the PDZ in the file
    """

    # Given
    expected_data = np.asarray(range(len(redshift_bins_fixture)), dtype=np.float32)

    # When
    with PdzDataProvider(pdz_data_files_fixture[1]) as provider:
        pos = provider.appendPdz(expected_data)

    # Then
    array = np.load(pdz_data_files_fixture[1])
    assert np.array_equal(array[pos, :], expected_data)


###############################################################################

def test_appendPdz_uninitialized(temp_dir_fixture, redshift_bins_fixture):
    """
    Test that calling appendPdz() to a file without header raises an exception
    """

    # Given
    filename = os.path.join(temp_dir_fixture, 'empty.npy')
    bad_data = np.asarray(range(len(redshift_bins_fixture)), dtype=np.float32)

    # When
    provider = PdzDataProvider(filename)

    # Then
    with pytest.raises(UninitializedException):
        provider.appendPdz(bad_data)


###############################################################################

def test_appendPdz_wrongLength(pdz_data_files_fixture, redshift_bins_fixture):
    """
    Test that appendPdz() raises an exception for wrong data length
    """

    # Given
    bad_data = np.asarray(range(len(redshift_bins_fixture) - 1), dtype=np.float32)

    # When
    provider = PdzDataProvider(pdz_data_files_fixture[1])

    # Then
    with pytest.raises(InvalidDimensionsException):
        provider.appendPdz(bad_data)


###############################################################################

def test_appendPdz_wrongDimensionality(pdz_data_files_fixture, redshift_bins_fixture):
    """
    Test that appendPdz() raises an exception for wrong data dimensionality
    """

    # Given
    bad_data = np.zeros((len(redshift_bins_fixture), 2), dtype=np.float32)

    # When
    provider = PdzDataProvider(pdz_data_files_fixture[1])

    # Then
    with pytest.raises(InvalidDimensionsException):
        provider.appendPdz(bad_data)


###############################################################################

def test_appendBulkPdz(pdz_data_files_fixture, redshift_bins_fixture):
    """
    Test that appendPdz() can append multiple pdz at once
    """

    # Given
    new_pdz = np.random.rand(5, len(redshift_bins_fixture))

    # When
    provider = PdzDataProvider(pdz_data_files_fixture[1])

    # Then
    offsets = provider.appendPdz(new_pdz)
    assert len(offsets) == len(new_pdz)
    for i, o in enumerate(offsets):
        pdz = provider.readPdz(o)
        assert np.allclose(new_pdz[i, :], pdz)
