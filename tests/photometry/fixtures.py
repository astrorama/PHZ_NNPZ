"""
Created on: 04/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import os
import pytest
import numpy as np
from astropy.table import Table, Column

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

@pytest.fixture()
def photometry_data_fixture():
    """Returns data for two photometry files.

    The data are the following:

    File photo1.fits:
    ID A1 A2 A3 A4
    -- -- -- -- --
     1  1  5  9 13
     3  2  6 10 14
    12  3  7 11 15
     7  4  8 12 16

    File photo2.fits:
    ID B1 B2 B3 A4
    -- -- -- -- --
     1 17 21 25 29
     3 18 22 26 30
    12 19 23 27 31
     7 20 24 28 32
    """
    return {
        'photo1.fits': {
            'ID': [1,3,12,7],
            'A1': [1,2,3,4],
            'A2': [5,6,7,8],
            'A3': [9,10,11,12],
            'A4': [13,14,15,16]
        },
        'photo2.fits': {
            'ID': [1,3,12,7],
            'B1': [17,18,19,20],
            'B2': [21,22,23,24],
            'B3': [25,26,27,28],
            'A4': [29,30,31,32]
        }
    }

##############################################################################

@pytest.fixture()
def photometry_dir_fixture(temp_dir_fixture, photometry_data_fixture):
    """Returns a directory which contains FITS files with photometry data"""
    for f in photometry_data_fixture:
        columns = photometry_data_fixture[f]
        t = Table()
        t.meta['EXTNAME'] = 'NNPZ_PHOTOMETRY'
        t['ID'] = Column(np.asarray(columns['ID'], dtype=np.int64))
        for name in columns:
            data = columns[name]
            if name != 'ID':
                t[name] = Column(np.asarray(data, dtype=np.float32))
        t.write(os.path.join(temp_dir_fixture, f), format='fits')
    return temp_dir_fixture

##############################################################################