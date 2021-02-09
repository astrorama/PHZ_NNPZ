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
Created on: 04/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import os
import pytest
import numpy as np
from astropy.table import Table, Column
import astropy.io.fits as fits

from nnpz.utils.fits import tableToHdu
from ..fixtures.util_fixtures import temp_dir_fixture

##############################################################################

@pytest.fixture()
def filters_fixture():
    """Returns the filters to be used for testing"""
    return [('First', np.asarray([(1, 0.1), (2, 0.2), (3, 0.4)], dtype=np.float32)),
            ('Second', np.asarray([(4, 0.4), (5, 0.5), (6, 0.6)], dtype=np.float32)),
            ('Third', np.asarray([(7, 0.7), (8, 0.8), (9, 0.9)], dtype=np.float32)),
            ('Fourth', np.asarray([(1, 0.11), (2, 0.22), (3, 0.44)], dtype=np.float32))
    ]

##############################################################################

@pytest.fixture()
def filter_dir_fixture(temp_dir_fixture, filters_fixture):
    """Returns a directory with filter transmissions.

    The directory contains the fiters from the filters_fixture in files with the
    extension "File.txt" (eg. FirstFile.txt, etc) and a filter_list.txt file
    defining their names and order.
    """

    filter_dir = os.path.join(temp_dir_fixture, 'filter_dir')
    os.makedirs(filter_dir)

    with open(os.path.join(filter_dir, 'filter_list.txt'), 'w') as list_file:
        for name, data in filters_fixture:
            list_file.write(name + 'File.dat : ' + name + '\n')
            with open(os.path.join(filter_dir, name + 'File.dat'), 'w') as f:
                for x, y in data:
                    f.write(str(x) + '\t' + str(y) + '\n')

    return filter_dir

##############################################################################

@pytest.fixture()
def filter_list_file_fixture(temp_dir_fixture, filters_fixture):
    """Returns a file containing a list with filter filenames.

    The list contains the filters from the filters_fixture stored in subdirectories
    following the order <Name>/<Name>File.txt. All the filters are aliased to their
    name.
    """

    filter_dir = os.path.join(temp_dir_fixture, 'filter_dir')
    os.makedirs(filter_dir)
    list_file = os.path.join(filter_dir, 'list_file.txt')
    with open(list_file, 'w') as lf:
        for name, data in filters_fixture:
            lf.write(name + '/' + name + 'File.dat : ' + name + '\n')
            f_dir = os.path.join(filter_dir, name)
            os.makedirs(f_dir)
            with open(os.path.join(f_dir, name + 'File.dat'), 'w') as f:
                for x, y in data:
                    f.write(str(x) + '\t' + str(y) + '\n')

    return list_file

##############################################################################