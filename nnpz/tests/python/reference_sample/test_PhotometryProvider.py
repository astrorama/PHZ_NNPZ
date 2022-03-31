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
Created on: 29/12/18
Author: Nikolaos Apostolakos
"""
from nnpz.exceptions import *
from nnpz.reference_sample import PhotometryProvider

from .fixtures import *


###############################################################################

def test_constructor_missingFile():
    """
    Test the constructor raises an exception if the file does not exist
    """

    # Given
    filename = '/does/not/exist.fits'

    # Then
    with pytest.raises(FileNotFoundError):
        PhotometryProvider(filename)


###############################################################################

def test_constructor_nonFitsFile(temp_dir_fixture: str):
    """Test the constructor with a non FITS file"""

    # Given
    filename = os.path.join(temp_dir_fixture, 'nonfits.txt')
    with open(filename, 'w'):
        pass

    # Then
    with pytest.raises(WrongFormatException):
        PhotometryProvider(filename)


###############################################################################

def test_constructor_missingNnpzHdu(photometry_file_fixture: str):
    """
    Test the case where the first table is not named NNPZ_PHOTOMETRY
    """

    # Given
    hdus = fits.open(photometry_file_fixture)
    hdus[1].header['EXTNAME'] = 'NOT_PHOT'
    os.unlink(photometry_file_fixture)
    hdus.writeto(photometry_file_fixture)

    # Then
    with pytest.raises(WrongFormatException):
        PhotometryProvider(photometry_file_fixture)


###############################################################################

def test_getType(photometry_file_fixture: str):
    """
    Test the getType method
    """

    # Given
    provider = PhotometryProvider(photometry_file_fixture)

    # When
    type = provider.get_type()

    # Then
    assert type == 'F_nu_uJy'


###############################################################################

def test_getFilterList(photometry_file_fixture: str, filters_fixture: Dict[str, np.ndarray]):
    """
    Test the getFilterList() method
    """

    # Given
    expected = set(filters_fixture.keys())

    # When
    provider = PhotometryProvider(photometry_file_fixture)
    filter_list = set(provider.get_filter_list())

    # Then
    assert filter_list == expected


###############################################################################

def test_getFilterTransmission_wrongName(photometry_file_fixture: str):
    """
    Test the getFilterTransmission with wrong filter name
    """

    # Given
    wrong_name = 'wrong'

    # When
    provider = PhotometryProvider(photometry_file_fixture)

    # Then
    with pytest.raises(KeyError):
        provider.get_filter_transmission(wrong_name)


###############################################################################

def test_getFilterTransmission_missingData(photometry_file_fixture: str):
    """
    Test the getFilterTransmission with filter for which the FITS has no data
    """

    # Given
    filter = 'vis'

    # When
    provider = PhotometryProvider(photometry_file_fixture)

    # Then
    with pytest.raises(MissingDataException):
        provider.get_filter_transmission(filter)


###############################################################################

def test_getFilterTransmission_success(photometry_file_fixture: str,
                                       filters_fixture: Dict[str, np.ndarray]):
    """
    Test successful call of getFilterTransmission()
    """

    # Given
    provider = PhotometryProvider(photometry_file_fixture)

    for name, expected in filters_fixture.items():
        if name == 'vis':
            continue

        # When
        trans = provider.get_filter_transmission(name)

        # Then
        np.testing.assert_array_equal(trans, expected)


###############################################################################

def test_getIds(photometry_file_fixture: str, photometry_ids_fixure: np.ndarray):
    """
    Test the getIds() method
    """

    # Given
    provider = PhotometryProvider(photometry_file_fixture)

    # When
    ids = provider.get_ids()

    # Then
    np.testing.assert_array_equal(ids, photometry_ids_fixure)


###############################################################################

def test_getData_noFilterList(photometry_file_fixture: str, photometry_data_fixture: np.ndarray):
    """
    Test that getData() returns all filters when called without argument
    """

    # When
    provider = PhotometryProvider(photometry_file_fixture)
    data = provider.get_data()
    filters = provider.get_filter_list()

    # Then
    np.testing.assert_array_equal(data[:, 0, :], photometry_data_fixture[filters[0]])
    np.testing.assert_array_equal(data[:, 1, :], photometry_data_fixture[filters[1]])
    np.testing.assert_array_equal(data[:, 2, :], photometry_data_fixture[filters[2]])

###############################################################################
