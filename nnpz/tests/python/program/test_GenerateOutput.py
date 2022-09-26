#
#  Copyright (C) 2022 Euclid Science Ground Segment
#
#  This library is free software; you can redistribute it and/or modify it under the terms of
#  the GNU Lesser General Public License as published by the Free Software Foundation;
#  either version 3.0 of the License, or (at your option) any later version.
#
#  This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
#  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License along with this library;
#  if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301 USA
#
import tempfile

import fitsio
import numpy as np
import pytest
from nnpz.program.GenerateOutput import patchOutputCatalog


@pytest.fixture
def conf_manager_fixture():
    return {
        'target_id_column': 'SOURCE_ID',
        'reference_ids': np.array([101, 202, 303, 404, 505, 606, 707, 808, 909])
    }


@pytest.fixture
def input_catalog_fixture():
    file = tempfile.NamedTemporaryFile(delete=False)
    print(file.name)
    fits = fitsio.FITS(file.name, mode='rw', clobber=True)
    data = np.zeros(2, dtype=[('SOURCE_ID', np.int64), ('NEIGHBOR_IDS', np.int64, 3),
                              ('NEIGHBOR_WEIGHTS', np.float32, 3),
                              ('NEIGHBOR_SCALING', np.float32, 3),
                              ('IGNORED', np.float32, 100),
                              ('FLAGS', np.int32)])

    data['SOURCE_ID'] = (np.arange(len(data)) + 1) * 10
    data['NEIGHBOR_IDS'][0] = [101, 303, 808]
    data['NEIGHBOR_IDS'][1] = [909, 505, 606]
    data['NEIGHBOR_WEIGHTS'] = np.random.randn(2, 3)
    data['NEIGHBOR_SCALING'] = np.random.randn(2, 3)
    data['FLAGS'] = [0, 2]
    fits.create_table_hdu(data)
    table = fits[1]
    table.write(data)  # create_table_hdu uses data only to populate the structure
    fits.close()
    return file, fitsio.FITS(file.name, mode='r')


def test_patchOutputCatalog(conf_manager_fixture, input_catalog_fixture):
    _, fits = input_catalog_fixture
    new_data, columns = patchOutputCatalog(conf_manager_fixture, fits[1])

    assert 'NEIGHBOR_INDEX' in columns
    assert 'NEIGHBOR_SCALING' in columns
    assert 'NEIGHBOR_WEIGHTS' in columns

    np.testing.assert_array_equal(new_data['NEIGHBOR_INDEX'][0], [0, 2, 7])
    np.testing.assert_array_equal(new_data['NEIGHBOR_INDEX'][1], [8, 4, 5])
    np.testing.assert_almost_equal(new_data['NEIGHBOR_WEIGHTS'], fits[1]['NEIGHBOR_WEIGHTS'][:])
    np.testing.assert_almost_equal(new_data['NEIGHBOR_SCALING'], fits[1]['NEIGHBOR_SCALING'][:])
    np.testing.assert_almost_equal(new_data['FLAGS'], fits[1]['FLAGS'][:])
