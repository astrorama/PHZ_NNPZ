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

import os
from typing import Dict

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from nnpz.photometry.colorspace import ColorSpace, RestFrame
from nnpz.photometry.photometric_system import PhotometricSystem
from nnpz.photometry.photometry import Photometry
from nnpz.reference_sample import PhotometryProvider

# noinspection PyUnresolvedReferences
from ..fixtures.util_fixtures import temp_dir_fixture


##############################################################################

@pytest.fixture()
def filters_fixture() -> Dict[str, np.ndarray]:
    """
    Returns the filters to be used for testing
    """
    filters = {'vis': np.asarray([
        (1000, 0.1),
        (1500, 1.0),
        (2000, 0.1),
        (2500, 0.0),
        (3000, 0.0),
        (3500, 0.0),
        (4000, 0.0),
        (4500, 0.0),
        (5000, 0.0)
    ], dtype=np.float32), 'g': np.asarray([
        (1000, 0.0),
        (1500, 0.0),
        (2000, 0.0),
        (2500, 0.0),
        (3000, 0.0),
        (3500, 0.0),
        (4000, 0.1),
        (4500, 1.0),
        (5000, 0.1)
    ], dtype=np.float32), 'Y': np.asarray([
        (1000, 0.0),
        (1500, 0.0),
        (2000, 0.0),
        (2500, 0.1),
        (3000, 1.0),
        (3500, 0.1),
        (4000, 0.0),
        (4500, 0.0),
        (5000, 0.0)
    ], dtype=np.float32)}
    return filters


##############################################################################

@pytest.fixture
def filter_dir_fixture(temp_dir_fixture: str, filters_fixture: Dict[str, np.ndarray]):
    """
    Returns a directory with filter transmissions.

    The directory contains the fiters from the filters_fixture in files with the
    extension "File.txt" (eg. FirstFile.txt, etc) and a filter_list.txt file
    defining their names and order.
    """

    filter_dir = os.path.join(temp_dir_fixture, 'filter_dir')
    os.makedirs(filter_dir)

    with open(os.path.join(filter_dir, 'filter_list.txt'), 'w') as list_file:
        for name, data in filters_fixture.items():
            list_file.write(name + 'File.dat : ' + name + '\n')
            with open(os.path.join(filter_dir, name + 'File.dat'), 'w') as f:
                for x, y in data:
                    f.write(str(x) + '\t' + str(y) + '\n')

    return filter_dir


##############################################################################

@pytest.fixture
def filter_list_file_fixture(temp_dir_fixture: str, filters_fixture: Dict[str, np.ndarray]) -> str:
    """
    Returns a file containing a list with filter filenames.

    The list contains the filters from the filters_fixture stored in subdirectories
    following the order <Name>/<Name>File.txt. All the filters are aliased to their
    name.
    """

    filter_dir = os.path.join(temp_dir_fixture, 'filter_dir')
    os.makedirs(filter_dir)
    list_file = os.path.join(filter_dir, 'list_file.txt')
    with open(list_file, 'w') as lf:
        for name, data in filters_fixture.items():
            lf.write(name + '/' + name + 'File.dat : ' + name + '\n')
            f_dir = os.path.join(filter_dir, name)
            os.makedirs(f_dir)
            with open(os.path.join(f_dir, name + 'File.dat'), 'w') as f:
                for x, y in data:
                    f.write(str(x) + '\t' + str(y) + '\n')

    return list_file


##############################################################################

@pytest.fixture
def reference_provider_fixture(temp_dir_fixture, filters_fixture) -> PhotometryProvider:
    photo_path = os.path.join(temp_dir_fixture, 'Photometry.fits')

    shift_corr = {
        'vis': np.zeros((5, 2), dtype=np.float32),
        'g': np.zeros((5, 2), dtype=np.float32),
        'Y': np.zeros((5, 2), dtype=np.float32),
    }
    shift_corr['vis'][:, 0] = np.arange(5, dtype=np.float32)
    shift_corr['g'][:, 1] = np.arange(5, dtype=np.float32)

    nnpz_photo = dict(ID=np.arange(5, dtype=np.int32))
    for i, filter_name in enumerate(filters_fixture.keys()):
        nnpz_photo[filter_name] = np.arange(1, 6, dtype=np.float32) * (i + 1) / 10.
        nnpz_photo[filter_name + '_SHIFT_CORR'] = shift_corr[filter_name]
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


##############################################################################

@pytest.fixture
def reference_photometry(reference_provider_fixture: PhotometryProvider) -> Photometry:
    return Photometry(
        reference_provider_fixture.get_ids(),
        reference_provider_fixture.get_data().astype(np.float64) * u.uJy,
        system=PhotometricSystem(reference_provider_fixture.get_filter_list()),
        colorspace=RestFrame)


##############################################################################

@pytest.fixture
def target_photometry(reference_photometry: Photometry) -> Photometry:
    """
    Generates a target catalog with E(B-V) and filter shifts per target.
    The filter shifts are generated from the filters_fixture
    """
    n_targets = 5
    shifts = np.zeros(5, dtype=[('vis', np.float32), ('g', np.float32), ('Y', np.float32)])
    shifts['vis'] = [0., 1500, 1500., 1500., 0.]
    shifts['g'] = [0., 0., 1000., 1000., 0.]
    shifts['Y'] = [0., 0., 0., -999., 0.]

    photo = np.zeros((n_targets, len(reference_photometry.system), 2), dtype=np.float64)
    for i in range(photo.shape[1]):
        photo[:, i, 0] = np.arange(1, 6, dtype=np.float64) * (
            i + 1) / 10. + [0.00698369, -0.01193469, 0.00892959, 0.00556434, 0.00172525]

    return Photometry(
        np.arange(1, n_targets + 1),
        values=photo * u.uJy,
        system=reference_photometry.system,
        colorspace=ColorSpace(ebv=np.zeros(n_targets), shifts=shifts)
    )

##############################################################################
