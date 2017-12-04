"""
Created on: 04/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import os
import pytest

from tests.util_fixtures import temp_dir_fixture

##############################################################################

@pytest.fixture()
def filters_fixture():
    """Returns the filters to be used for testing"""
    return [('First', [(1, 0.1), (2, 0.2), (3, 0.4)]),
            ('Second', [(4, 0.4), (5, 0.5), (6, 0.6)]),
            ('Third', [(7, 0.7), (8, 0.8), (9, 0.9)]),
            ('Fourth', [(1, 0.11), (2, 0.22), (3, 0.44)])
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
            list_file.write(name + 'File.txt : ' + name + '\n')
            with open(os.path.join(filter_dir, name + 'File.txt'), 'w') as f:
                for x, y in data:
                    f.write(str(x) + '\t' + str(y) + '\n')

    return filter_dir

##############################################################################