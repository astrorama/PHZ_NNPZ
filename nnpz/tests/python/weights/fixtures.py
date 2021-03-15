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
Created: 11/04/2018
Author: Alejandro Alvarez Ayllon
"""

from __future__ import division, print_function

import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from nnpz.reference_sample import PhotometryProvider
from nnpz.reference_sample.ReferenceSample import ReferenceSample

# noinspection PyUnresolvedReferences
from ..fixtures.util_fixtures import temp_dir_fixture


@pytest.fixture
def reference_sample_fixture(temp_dir_fixture):
    """
    Generates a reference sample to be used by the weight tests
    """
    n_references = 5

    # Use this SED as a sample
    test_sed_file = os.path.join(os.path.dirname(__file__), 'test.sed')
    test_sed_table = Table.read(test_sed_file, format='ascii')
    test_sed = np.asarray((test_sed_table['col1'], test_sed_table['col2'])).transpose()

    os.rmdir(temp_dir_fixture)
    ref_sample = ReferenceSample.createNew(temp_dir_fixture)
    for obj_id in range(1, n_references + 1):
        # Generate a SED shifted by a random Z
        z = obj_id
        new_sed = np.ndarray(test_sed.shape)
        new_sed[:, 0] = test_sed[:, 0] * z
        new_sed[:, 1] = test_sed[:, 1] / z
        ref_sample.getProvider('sed').addData([obj_id], new_sed.reshape(1, -1, 2))
        # Can skip the PDZ for these tests

    return ref_sample


@pytest.fixture
def filters_fixture():
    """
    Returns the filters to be used for testing
    """
    filters = {}
    filters['vis'] = np.asarray([
        (1000, 0.1),
        (1500, 1.0),
        (2000, 0.1),
        (2500, 0.0),
        (3000, 0.0),
        (3500, 0.0),
        (4000, 0.0),
        (4500, 0.0),
        (5000, 0.0)
    ], dtype=np.float32)
    filters['g'] = np.asarray([
        (1000, 0.0),
        (1500, 0.0),
        (2000, 0.0),
        (2500, 0.0),
        (3000, 0.0),
        (3500, 0.0),
        (4000, 0.1),
        (4500, 1.0),
        (5000, 0.1)
    ], dtype=np.float32)
    filters['Y'] = np.asarray([
        (1000, 0.0),
        (1500, 0.0),
        (2000, 0.0),
        (2500, 0.1),
        (3000, 1.0),
        (3500, 0.1),
        (4000, 0.0),
        (4500, 0.0),
        (5000, 0.0)
    ], dtype=np.float32)

    return filters


@pytest.fixture
def target_fixture(filters_fixture):
    """
    Generates a target catalog with E(B-V) and filter shifts per target.
    The filter shifts are generated from the filters_fixture
    """
    filter_means = {}
    trans_avg = {}
    for filter_name, transmissions in filters_fixture.items():
        trans_avg[filter_name] = np.average(transmissions[:, 0], weights=transmissions[:, 1])

    filter_means['vis'] = trans_avg['vis'] + np.array([0., 1500, 1500., 1500., np.nan])
    filter_means['g'] = trans_avg['g'] + np.array([0., 0., 1000., 1000., np.nan])
    filter_means['Y'] = trans_avg['Y'] + np.array([0., 0., 0., -999., np.nan])

    n_targets = len(filter_means['vis'])
    return {
        'ID': np.arange(1, n_targets + 1),
        'ebv': np.zeros(n_targets),
        'filter_means': filter_means,
    }


@pytest.fixture
def reference_photo_fixture(temp_dir_fixture, filters_fixture):
    photo_path = os.path.join(temp_dir_fixture, 'Photometry.fits')

    nnpz_photo = dict(ID=np.arange(5, dtype=np.int))
    for filter_name in filters_fixture.keys():
        nnpz_photo[filter_name] = np.zeros(5, dtype=np.float32)
    nnpz_photo = fits.BinTableHDU(
        data=Table(nnpz_photo),
        name='NNPZ_PHOTOMETRY',
        header=dict(PHOTYPE='F_nu_uJy')
    )

    hdus = [fits.PrimaryHDU(), nnpz_photo]

    for filter_name, transmission in filters_fixture.items():
        hdus.append(fits.BinTableHDU(data=Table(dict(
            Wavelength=transmission[:, 0],
            Transmission=transmission[:, 1]
        )), name=filter_name))

    fits.HDUList(hdus).writeto(photo_path)
    return PhotometryProvider(photo_path)
