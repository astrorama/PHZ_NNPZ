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
Created on: 04/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import os
import pytest
import numpy as np
from astropy.table import Table, Column
import astropy.io.fits as fits

from nnpz.utils import Fits
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
            list_file.write(name + 'File.txt : ' + name + '\n')
            with open(os.path.join(filter_dir, name + 'File.txt'), 'w') as f:
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
            lf.write(name + '/' + name + 'File.txt : ' + name + '\n')
            f_dir = os.path.join(filter_dir, name)
            os.makedirs(f_dir)
            with open(os.path.join(f_dir, name + 'File.txt'), 'w') as f:
                for x, y in data:
                    f.write(str(x) + '\t' + str(y) + '\n')

    return list_file

##############################################################################

@pytest.fixture()
def photometry_ids_fixure():
    """Returns the IDs for the photometry file"""
    return range(1, 100)

##############################################################################

@pytest.fixture()
def photometry_data_fixture(photometry_ids_fixure, filters_fixture):
    """Returns the data for the photometry file.

    The data are random numbers for each filter, both for the value and the
    error. Only for the second filter the errors are zeros, indicating a missing
    error column.
    """
    data = np.random.rand(len(photometry_ids_fixure), len(filters_fixture), 2).astype(np.float32)
    data[:,1,1] = 0
    return data

##############################################################################

@pytest.fixture()
def photometry_file_fixture(temp_dir_fixture, photometry_ids_fixure, filters_fixture, photometry_data_fixture):
    """Returns the path of a FITS file which contains the photometry.

    All filters contain errors except of the second one. The transmission of the
    third filter is not stored in the file. The type of the photometry is set to
    F_nu_uJy.
    """

    hdus = fits.HDUList()

    t = Table()
    t.meta['EXTNAME'] = 'NNPZ_PHOTOMETRY'
    t.meta['PHOTYPE'] = 'F_nu_uJy'

    t['ID'] = Column(np.asarray(photometry_ids_fixure, dtype=np.int64))

    for i, filter in enumerate(filters_fixture):
        name = filter[0]
        t[name] = Column(photometry_data_fixture[:, i, 0])
        if not np.all(photometry_data_fixture[:, i, 1] == 0):
            t[name+'_ERR'] = Column(photometry_data_fixture[:, i, 1])

    hdus.append(Fits.tableToHdu(t))

    for name, data in filters_fixture:
        if name == filters_fixture[3][0]:
            continue
        t = Table()
        t.meta['EXTNAME'] = name
        t['Wavelength'] = Column(data[:, 0])
        t['Transmission'] = Column(data[:, 1])
        hdus.append(Fits.tableToHdu(t))

    filename = os.path.join(temp_dir_fixture, 'phot.fits')
    hdus.writeto(filename)
    return filename

##############################################################################
