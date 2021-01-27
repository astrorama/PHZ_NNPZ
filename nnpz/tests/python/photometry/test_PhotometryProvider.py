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
Created on: 29/12/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import pytest
import os

from nnpz.exceptions import *
from nnpz.photometry import PhotometryProvider
from .fixtures import *


###############################################################################

def test_constructor_missingFile():
    """Test the constructor raises an exception if the file does not exist"""

    # Given
    filename = '/does/not/exist.fits'

    # Then
    with pytest.raises(FileNotFoundException):
        PhotometryProvider(filename)

###############################################################################

def test_constructor_nonFitsFile(temp_dir_fixture):
    """Test the constructor with a non FITS file"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'nonfits.txt')
    with open(filename,'w'):
        pass

    # Then
    with pytest.raises(WrongFormatException):
        PhotometryProvider(filename)

###############################################################################

def test_constructor_missingNnpzHdu(photometry_file_fixture):
    """Test the case where the first table is not named NNPZ_PHOTOMETRY"""

    # Given
    hdus = fits.open(photometry_file_fixture)
    hdus[1].header['EXTNAME'] = 'NOT_PHOT'
    os.unlink(photometry_file_fixture)
    hdus.writeto(photometry_file_fixture)

    # Then
    with pytest.raises(WrongFormatException):
        PhotometryProvider(photometry_file_fixture)

###############################################################################

def test_getType(photometry_file_fixture):
    """Test the getType method"""

    # Given
    provider = PhotometryProvider(photometry_file_fixture)

    # When
    type = provider.getType()

    # Then
    assert type == 'F_nu_uJy'

###############################################################################

def test_getFilterList(photometry_file_fixture, filters_fixture):
    """Test the getFilterList() method"""

    # Given
    expected = [f for f,d in filters_fixture]

    # When
    provider = PhotometryProvider(photometry_file_fixture)
    filter_list = provider.getFilterList()

    # Then
    assert filter_list == expected

###############################################################################

def test_getFilterTransmission_wrongName(photometry_file_fixture):
    """Test the getFilterTransmission with wrong filter name"""

    # Given
    wrong_name = 'wrong'

    # When
    provider = PhotometryProvider(photometry_file_fixture)

    # Then
    with pytest.raises(UnknownNameException):
        provider.getFilterTransmission(wrong_name)

###############################################################################

def test_getFilterTransmission_missingData(photometry_file_fixture, filters_fixture):
    """Test the getFilterTransmission with filter for which the FITS has no data"""

    # Given
    filter = filters_fixture[3][0]

    # When
    provider = PhotometryProvider(photometry_file_fixture)

    # Then
    with pytest.raises(MissingDataException):
        provider.getFilterTransmission(filter)

###############################################################################

def test_getFilterTransmission_success(photometry_file_fixture, filters_fixture):
    """Test successful call of getFilterTransmission()"""

    # Given
    provider = PhotometryProvider(photometry_file_fixture)

    for name, expected in filters_fixture:
        if name == filters_fixture[3][0]:
            continue

        # When
        trans = provider.getFilterTransmission(name)

        # Then
        assert np.array_equal(trans, expected)

###############################################################################

def test_getIds(photometry_file_fixture, photometry_ids_fixure):
    """Test the getIds() method"""

    # Given
    provider = PhotometryProvider(photometry_file_fixture)

    # When
    ids = provider.getIds()

    # Then
    assert np.array_equal(ids, photometry_ids_fixure)

###############################################################################

def test_getData_wrongFilter(photometry_file_fixture):
    """Test the getData() with a filter name that is not in the file"""

    # Given
    wrong_name = 'wrong_name'

    # When
    provider = PhotometryProvider(photometry_file_fixture)

    # Then
    with pytest.raises(UnknownNameException):
        provider.getData(wrong_name)

###############################################################################

def test_getData_noFilterList(photometry_file_fixture, photometry_data_fixture):
    """Test that getData() returns all filters when called without argument"""

    # When
    provider = PhotometryProvider(photometry_file_fixture)
    data = provider.getData()

    # Then
    assert np.array_equal(data, photometry_data_fixture)

###############################################################################

def test_getData_withArgs(photometry_file_fixture, photometry_data_fixture, filters_fixture):
    """Test the getData() call for only some of the filters"""

    # Given
    indices = [2, 1]
    filters = [filters_fixture[i][0] for i in indices]
    expected = np.zeros((len(photometry_data_fixture), len(filters), 2), dtype=np.float32)
    for expected_i, all_i in enumerate(indices):
        expected[:,expected_i,:] = photometry_data_fixture[:,all_i,:]

    # When
    provider = PhotometryProvider(photometry_file_fixture)
    data = provider.getData(*filters)

    # Then
    assert np.array_equal(data, expected)

###############################################################################
