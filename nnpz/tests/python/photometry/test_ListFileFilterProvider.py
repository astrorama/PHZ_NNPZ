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
Created on: 18/12/17
Author: Nikolaos Apostolakos
"""

from nnpz.exceptions import *
from nnpz.photometry.filter_provider.list_file_filter_provider import ListFileFilterProvider

from .fixtures import *


###############################################################################

def test_constructor_wrongPath():
    """
    Test the case where the given path does not exist
    """

    # Given
    wrong_path = '/wrong/path/file.txt'

    # Then
    with pytest.raises(FileNotFoundError):
        ListFileFilterProvider(wrong_path)


###############################################################################

def test_constructor_missingFilterFile(filter_list_file_fixture: str):
    """
    Test the case where there is a file declared in list file which is missing
    """

    # Given
    with open(filter_list_file_fixture, 'a') as f:
        f.write('MissingFile.txt : Missing')

    # Then
    with pytest.raises(FileNotFoundError):
        ListFileFilterProvider(filter_list_file_fixture)


###############################################################################

def test_getFilterNames(filter_list_file_fixture: str, filters_fixture: Dict[str, np.ndarray]):
    """
    Test the filter names from the list file, when names are defined
    """

    # Given
    expected_names = list(filters_fixture.keys())

    # When
    provider = ListFileFilterProvider(filter_list_file_fixture)
    names = provider.get_filter_names()

    # Then
    assert len(names) == len(expected_names)
    assert names == expected_names


###############################################################################

def test_getFilterNames_undefinedName(filter_list_file_fixture: str,
                                      filters_fixture: Dict[str, np.ndarray]):
    """
    Test the case where the filter list contains entries without the name
    """

    # Given
    with open(filter_list_file_fixture) as f:
        lines = f.readlines()
    lines[1] = lines[1][:lines[1].index(':')] + '\n'
    with open(filter_list_file_fixture, 'w') as f:
        for l in lines:
            f.write(l)
    expected_names = list(filters_fixture.keys())
    expected_names[1] = expected_names[1] + 'File'

    # When
    provider = ListFileFilterProvider(filter_list_file_fixture)
    names = provider.get_filter_names()

    # Then
    assert len(names) == len(expected_names)
    assert names == expected_names


###############################################################################

def test_getFilterNames_absolutePath(temp_dir_fixture: str, filter_list_file_fixture: str,
                                     filters_fixture: Dict[str, np.ndarray]):
    """
    Test the case where the filter list contains an absolute path
    """

    # Given
    abs_filter_file = os.path.abspath(os.path.join(temp_dir_fixture, 'abs.txt'))
    with open(abs_filter_file, 'w') as f:
        for i in range(10):
            f.write(str(i) + '\t' + str(i) + '\n')

    with open(filter_list_file_fixture) as f:
        lines = f.readlines()
    lines.append(abs_filter_file + '\n')
    with open(filter_list_file_fixture, 'w') as f:
        for l in lines:
            f.write(l)
    expected_names = list(filters_fixture.keys()) + ['abs']

    # When
    provider = ListFileFilterProvider(filter_list_file_fixture)
    names = provider.get_filter_names()

    # Then
    assert len(names) == len(expected_names)
    assert names == expected_names


###############################################################################

def test_getFilterTransmission_unknownName(filter_list_file_fixture: str):
    """
    Test the getFilterTransmission with a wrong filter name
    """

    # Given
    wrong_name = 'wrong_name'

    # When
    provider = ListFileFilterProvider(filter_list_file_fixture)

    # Then
    with pytest.raises(UnknownNameException):
        provider.get_filter_transmission(wrong_name)


###############################################################################

def test_getFilterTransmission_success(filter_list_file_fixture: str,
                                       filters_fixture: Dict[str, np.ndarray]):
    """
    Test the getFilterTransmission() successful call
    """

    # Given
    provider = ListFileFilterProvider(filter_list_file_fixture)

    for name, expected_data in filters_fixture.items():
        # When
        data = provider.get_filter_transmission(name)

        # Then
        assert np.array_equal(data, np.asarray(expected_data, dtype=np.float32))


###############################################################################

def test_getFilterTransmission_success_absolutePath(temp_dir_fixture: str,
                                                    filter_list_file_fixture: str):
    """
    Test the getFilterTransmission() successful call for an absolute path
    """

    # Given
    abs_filter_file = os.path.abspath(os.path.join(temp_dir_fixture, 'abs.txt'))
    expected_data = np.asarray([(1, 2), (3, 4), (5, 6)], dtype=np.float32)
    with open(abs_filter_file, 'w') as f:
        for x, y in expected_data:
            f.write(str(x) + '\t' + str(y) + '\n')

    with open(filter_list_file_fixture) as f:
        lines = f.readlines()
    lines.append(abs_filter_file + '\n')
    with open(filter_list_file_fixture, 'w') as f:
        for l in lines:
            f.write(l)

    # When
    provider = ListFileFilterProvider(filter_list_file_fixture)
    data = provider.get_filter_transmission('abs')

    # Then
    assert np.array_equal(data, expected_data)

###############################################################################
