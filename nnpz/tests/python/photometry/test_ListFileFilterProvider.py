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
Created on: 18/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import pytest
import os
import numpy as np

from nnpz.exceptions import *
from nnpz.photometry import ListFileFilterProvider

from .fixtures import filters_fixture, filter_list_file_fixture, temp_dir_fixture

###############################################################################

def test_constructor_wrongPath():
    """Test the case where the given path does not exist"""

    # Given
    wrong_path = '/wrong/path/file.txt'

    # Then
    with pytest.raises(FileNotFoundException):
        ListFileFilterProvider(wrong_path)

###############################################################################

def test_constructor_missingFilterFile(filter_list_file_fixture):
    """Test the case where there is a file declared in list file which is missing"""

    # Given
    with open(filter_list_file_fixture, 'a') as f:
        f.write('MissingFile.txt : Missing')

    # Then
    with pytest.raises(FileNotFoundException):
        ListFileFilterProvider(filter_list_file_fixture)

###############################################################################

def test_getFilterNames(filter_list_file_fixture, filters_fixture):
    """Test the filter names from the list file, when names are defined"""

    # Given
    expected_names = [name for name,_ in filters_fixture]

    # When
    provider = ListFileFilterProvider(filter_list_file_fixture)
    names = provider.getFilterNames()

    # Then
    assert len(names) == len(expected_names)
    assert names == expected_names

###############################################################################

def test_getFilterNames_undefinedName(filter_list_file_fixture, filters_fixture):
    """Test the case where the filter list contains entries without the name"""

    # Given
    with open(filter_list_file_fixture) as f:
        lines = f.readlines()
    lines[1] = lines[1][:lines[1].index(':')] + '\n'
    with open(filter_list_file_fixture, 'w') as f:
        for l in lines:
            f.write(l)
    expected_names = [name for name,_ in filters_fixture]
    expected_names[1] = expected_names[1] + 'File'

    # When
    provider = ListFileFilterProvider(filter_list_file_fixture)
    names = provider.getFilterNames()

    # Then
    assert len(names) == len(expected_names)
    assert names == expected_names

###############################################################################

def test_getFilterNames_absolutePath(temp_dir_fixture, filter_list_file_fixture, filters_fixture):
    """Test the case where the filter list contains an absolute path"""

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
    expected_names = [name for name,_ in filters_fixture] + ['abs']

    # When
    provider = ListFileFilterProvider(filter_list_file_fixture)
    names = provider.getFilterNames()

    # Then
    assert len(names) == len(expected_names)
    assert names == expected_names


###############################################################################

def test_getFilterTransmission_unknownName(filter_list_file_fixture):
    """Test the getFilterTransmission with a wrong filter name"""

    # Given
    wrong_name = 'wrong_name'

    # When
    provider = ListFileFilterProvider(filter_list_file_fixture)

    # Then
    with pytest.raises(UnknownNameException):
        provider.getFilterTransmission(wrong_name)

###############################################################################

def test_getFilterTransmission_success(filter_list_file_fixture, filters_fixture):
    """Test the getFilterTransmission() successful call"""

    # Given
    provider = ListFileFilterProvider(filter_list_file_fixture)

    for name, expected_data in filters_fixture:

        # When
        data = provider.getFilterTransmission(name)

        # Then
        assert np.array_equal(data, np.asarray(expected_data, dtype=np.float32))

###############################################################################

def test_getFilterTransmission_success_absolutePath(temp_dir_fixture, filter_list_file_fixture, filters_fixture):
    """Test the getFilterTransmission() successful call for an absolute path"""

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
    data = provider.getFilterTransmission('abs')

    # Then
    assert np.array_equal(data, expected_data)

###############################################################################

