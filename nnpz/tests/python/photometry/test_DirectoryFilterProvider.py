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
Created on: 04/12/17
Author: Nikolaos Apostolakos
"""
from nnpz.exceptions import *
from nnpz.photometry.filter_provider.directory_filter_provider import DirectoryFilterProvider

from .fixtures import *


###############################################################################

def test_constructor_wrongPath(temp_dir_fixture: str):
    """
    Test the case where the given path is not a directory
    """

    # Given
    wrong_path = os.path.join(temp_dir_fixture, 'wrong_path')

    # Then
    with pytest.raises(NotADirectoryError):
        DirectoryFilterProvider(wrong_path)


###############################################################################

def test_constructor_missingFilterFile(filter_dir_fixture: str):
    """
    Test the case where there is a file declared in filter_list.txt which is missing
    """

    # Given
    with open(os.path.join(filter_dir_fixture, 'filter_list.txt'), 'a') as f:
        f.write('MissingFile.txt : Missing')

    # Then
    with pytest.raises(FileNotFoundError):
        DirectoryFilterProvider(filter_dir_fixture)


###############################################################################

def test_getFilterNames_noFilterListFile(filter_dir_fixture: str,
                                         filters_fixture: Dict[str, np.ndarray]):
    """
    Test the filter names when there is no filter_list.tx
    """

    # Given
    os.remove(os.path.join(filter_dir_fixture, 'filter_list.txt'))
    expected_names = [name + 'File' for name in filters_fixture.keys()]

    # When
    provider = DirectoryFilterProvider(filter_dir_fixture)
    names = provider.get_filter_names()

    # Then
    assert len(names) == len(expected_names)
    for n in names:
        assert n in expected_names


###############################################################################

def test_getFilterNames_filterListFile(filter_dir_fixture: str,
                                       filters_fixture: Dict[str, np.ndarray]):
    """
    Test the filter names from the filter_list.txt, when names are defined
    """

    # Given
    expected_names = [name for name in filters_fixture.keys()]

    # When
    provider = DirectoryFilterProvider(filter_dir_fixture)
    names = provider.get_filter_names()

    # Then
    assert len(names) == len(expected_names)
    assert names == expected_names


###############################################################################

def test_getFilterNames_filterListFile_extraFilterFile(filter_dir_fixture: str,
                                                       filters_fixture: Dict[str, np.ndarray]):
    """
    Test the filter names from the filter_list.txt, when names are defined
    """

    # Given
    with open(os.path.join(filter_dir_fixture, 'extra.txt'), 'w') as f:
        for i in range(10):
            f.write(str(i) + '\t' + str(i) + '\n')
    expected_names = [name for name in filters_fixture.keys()]

    # When
    provider = DirectoryFilterProvider(filter_dir_fixture)
    names = provider.get_filter_names()

    # Then
    assert len(names) == len(expected_names)
    assert names == expected_names


###############################################################################

def test_getFilterNames_filterListFile_undefinedName(filter_dir_fixture: str,
                                                     filters_fixture: Dict[str, np.ndarray]):
    """
    Test the case where the filter_list.txt contains entries without the name
    """

    # Given
    with open(os.path.join(filter_dir_fixture, 'filter_list.txt')) as f:
        lines = f.readlines()
    lines[1] = lines[1][:lines[1].index(':')] + '\n'
    with open(os.path.join(filter_dir_fixture, 'filter_list.txt'), 'w') as f:
        for l in lines:
            f.write(l)
    expected_names = [name for name in filters_fixture.keys()]
    expected_names[1] = expected_names[1] + 'File'

    # When
    provider = DirectoryFilterProvider(filter_dir_fixture)
    names = provider.get_filter_names()

    # Then
    assert len(names) == len(expected_names)
    assert names == expected_names


###############################################################################

def test_getFilterTransmission_unknownName(filter_dir_fixture: str):
    """
    Test the getFilterTransmission with a wrong filter name
    """

    # Given
    wrong_name = 'wrong_name'

    # When
    provider = DirectoryFilterProvider(filter_dir_fixture)

    # Then
    with pytest.raises(UnknownNameException):
        provider.get_filter_transmission(wrong_name)


###############################################################################

def test_getFilterTransmission_success(filter_dir_fixture: str,
                                       filters_fixture: Dict[str, np.ndarray]):
    """
    Test the getFilterTransmission() successful call
    """

    # Given
    provider = DirectoryFilterProvider(filter_dir_fixture)

    for name, expected_data in filters_fixture.items():
        # When
        data = provider.get_filter_transmission(name)

        # Then
        assert np.array_equal(data, np.asarray(expected_data, dtype=np.float32))

###############################################################################
