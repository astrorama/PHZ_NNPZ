"""
Created: 23/09/2019
Author: Alejandro Alvarez Ayllon
"""

from __future__ import division, print_function

import numpy as np
import pytest
from nnpz.flags import NnpzFlag
from nnpz.io.catalog_properties import Photometry
from astropy.table import Table


@pytest.fixture()
def photometry_cat_fixture():
    return Table({
        'FLUX_G': [1958., 6352., 36845., -1., 55554.],
        'FLUX_G_ERR': [10., 20., 30., -1., 55.],
        'FLUX_Y': [800., 900., 555., 888., -1.],
        'FLUX_Y_ERR': [20., 30., 10., 22., -1.],
        'FLUX_G_CORRECTION': [1, 2, 3, 1, 2],
        'FLUX_Y_CORRECTION': [2, 4, 8, 3, 5],
    })


###############################################################################

def test_onlyPhotometry(photometry_cat_fixture):
    """
    Read only the photometry column
    """
    # Given
    filters = [('FLUX_G', None), ('FLUX_Y', None)]
    photometry_reader = Photometry(filters, nan_flags=[-1])
    photometry = photometry_reader(photometry_cat_fixture)

    # Dimensions match the requested data
    assert (len(photometry.shape) == 3)
    assert (photometry.shape[0] == len(photometry_cat_fixture))
    assert (photometry.shape[1] == len(filters))
    assert (photometry.shape[2] == 2)

    # No error present
    assert (np.all(photometry[:, :, 1] == 0.))

    # Values kept
    g_nan = photometry_cat_fixture['FLUX_G'] == -1
    y_nan = photometry_cat_fixture['FLUX_Y'] == -1

    assert ((photometry[:, 0, 0] == photometry_cat_fixture['FLUX_G'])[g_nan == False].all())
    assert ((photometry[:, 1, 0] == photometry_cat_fixture['FLUX_Y'])[y_nan == False].all())

    # -1 turned into NaN
    assert (np.isnan(photometry[:, 0, 0][g_nan]).all())
    assert (np.isnan(photometry[:, 1, 0][y_nan]).all())


###############################################################################


def test_photometryAndError(photometry_cat_fixture):
    """
    Read photometry and error columns
    """
    # Given
    filters = [('FLUX_G', 'FLUX_G_ERR'), ('FLUX_Y', 'FLUX_Y_ERR')]
    photometry_reader = Photometry(filters, nan_flags=[-1])
    photometry = photometry_reader(photometry_cat_fixture)

    # Dimensions match the requested data
    assert (len(photometry.shape) == 3)
    assert (photometry.shape[0] == len(photometry_cat_fixture))
    assert (photometry.shape[1] == len(filters))
    assert (photometry.shape[2] == 2)

    # Values kept
    g_nan = photometry_cat_fixture['FLUX_G'] == -1
    y_nan = photometry_cat_fixture['FLUX_Y'] == -1

    assert ((photometry[:, 0, 0] == photometry_cat_fixture['FLUX_G'])[g_nan == False].all())
    assert ((photometry[:, 0, 1] == photometry_cat_fixture['FLUX_G_ERR'])[g_nan == False].all())
    assert ((photometry[:, 1, 0] == photometry_cat_fixture['FLUX_Y'])[y_nan == False].all())
    assert ((photometry[:, 1, 1] == photometry_cat_fixture['FLUX_Y_ERR'])[y_nan == False].all())

    # -1 turned into NaN
    assert (np.isnan(photometry[:, 0, 0][g_nan]).all())
    assert (np.isnan(photometry[:, 0, 1][g_nan]).all())
    assert (np.isnan(photometry[:, 1, 0][y_nan]).all())
    assert (np.isnan(photometry[:, 1, 1][y_nan]).all())


###############################################################################

def test_photometryErrorAndCorrection(photometry_cat_fixture):
    """
    Read photometry, error and correction
    """
    # Given
    filters = [('FLUX_G', 'FLUX_G_ERR', 'FLUX_G_CORRECTION'), ('FLUX_Y', 'FLUX_Y_ERR', 'FLUX_Y_CORRECTION')]
    photometry_reader = Photometry(filters, nan_flags=[-1])
    photometry = photometry_reader(photometry_cat_fixture)

    # Dimensions match the requested data
    assert (len(photometry.shape) == 3)
    assert (photometry.shape[0] == len(photometry_cat_fixture))
    assert (photometry.shape[1] == len(filters))
    assert (photometry.shape[2] == 2)

    # Values kept with correction applied, errors unaffected
    g_nan = photometry_cat_fixture['FLUX_G'] == -1
    y_nan = photometry_cat_fixture['FLUX_Y'] == -1

    expected_g = photometry_cat_fixture['FLUX_G'] * photometry_cat_fixture['FLUX_G_CORRECTION']
    expected_y = photometry_cat_fixture['FLUX_Y'] * photometry_cat_fixture['FLUX_Y_CORRECTION']
    assert ((photometry[:, 0, 0] == expected_g)[g_nan == False].all())
    assert ((photometry[:, 0, 1] == photometry_cat_fixture['FLUX_G_ERR'])[g_nan == False].all())
    assert ((photometry[:, 1, 0] == expected_y)[y_nan == False].all())
    assert ((photometry[:, 1, 1] == photometry_cat_fixture['FLUX_Y_ERR'])[y_nan == False].all())

    # -1 turned into NaN
    assert (np.isnan(photometry[:, 0, 0][g_nan]).all())
    assert (np.isnan(photometry[:, 0, 1][g_nan]).all())
    assert (np.isnan(photometry[:, 1, 0][y_nan]).all())
    assert (np.isnan(photometry[:, 1, 1][y_nan]).all())
