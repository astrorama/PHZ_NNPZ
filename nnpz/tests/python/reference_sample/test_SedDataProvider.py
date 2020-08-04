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

from nnpz.exceptions import *
from nnpz.reference_sample.SedDataProvider import SedDataProvider

from .fixtures import *


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
        found_data = provider.readSed(pos)

        # Then
        assert np.array_equal(found_data, expected_data)

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
        pos = provider.appendSed(expected_data)

    # Then
    with SedDataProvider(sed_data_files_fixture[1]) as provider:
        data = provider.readSed(pos)
        np.array_equal(expected_data, data)


###############################################################################

def test_appendSed_invalidDataDimension(sed_data_files_fixture):
    """
    Test that the appendSed() method raises an exception for wrong data dimensions
    """

    # Given
    three_dim_data = np.zeros((100, 2, 2), dtype=np.float32)
    wrong_second_dimension = np.zeros((100, 3), dtype=np.float32)

    # When
    provider = SedDataProvider(sed_data_files_fixture[1])

    # Then
    with pytest.raises(InvalidDimensionsException):
        provider.appendSed(three_dim_data)
    with pytest.raises(InvalidDimensionsException):
        provider.appendSed(wrong_second_dimension)


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
        provider.appendSed(wrong_wavelength)
