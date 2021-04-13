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
import numpy as np
import pytest
from astropy.table import Column, Table
from nnpz.framework.NeighborSet import NeighborSet
from nnpz.io.output_column_providers.UniformPhotometry import UniformPhotometry

# noinspection PyUnresolvedReferences
from .fixtures import mock_output_handler


class DummyPhotometry(object):
    def __init__(self):
        self._data = np.zeros((2, 2), dtype=[('A', np.float32), ('B', np.float32), ('C', np.float32)])
        self._data['A'][:, 0] = [0.5626045, 0.94242]
        self._data['B'][:, 0] = [1.7679665, 1.8930919]
        self._data['C'][:, 0] = [3.5187345, 2.606762]

    def getData(self, *filter_list):
        data = np.zeros((len(self._data), len(filter_list), 2), dtype=np.float32)
        for i, f in enumerate(filter_list):
            data[:, i, 0] = self._data[f][:, 0]
        return data


@pytest.fixture
def reference_photometry():
    return DummyPhotometry()


@pytest.fixture()
def reference_matched_photometry(reference_photometry):
    matched = reference_photometry._data.copy()
    matched['A'][:, 0] *= np.array([1., 2.])
    matched['B'][:, 0] *= np.array([1.5, 1.7])
    matched['C'][:, 0] *= np.array([1.2, 1.5])
    return matched


@pytest.fixture
def catalog_photometry():
    return Table([
        Column([10], name='FLUX_A'), Column([0], name='FLUX_A_ERR'),
        Column([10], name='FLUX_B'), Column([0], name='FLUX_B_ERR'),
        Column([10], name='FLUX_C'), Column([0], name='FLUX_C_ERR'),
    ])


def test_uniform_photometry(reference_photometry, reference_matched_photometry, catalog_photometry,
                            mock_output_handler):
    uniform = UniformPhotometry(
        catalog_photometry, reference_photometry, {
            ('MY_A_A', 'MY_A_A_ERR'): ('A', 'A', 'FLUX_A', 'FLUX_A_ERR'),
            ('MY_B_B', 'MY_B_B_ERR'): ('B', 'B', 'FLUX_B', 'FLUX_B_ERR'),
            ('MY_C_C', 'MY_C_C_ERR'): ('C', 'C', 'FLUX_C', 'FLUX_C_ERR'),
            ('MY_C_A', 'MY_C_A_ERR'): ('C', 'A', 'FLUX_A', 'FLUX_A_ERR')
        }
    )
    mock_output_handler.addColumnProvider(uniform)
    mock_output_handler.initialize(len(catalog_photometry))

    ns = NeighborSet()
    [ns.append(0) for _ in range(len(reference_matched_photometry))]

    for i, (t, p) in enumerate(zip(ns, reference_matched_photometry)):
        t.weight = 1.
        t.matched_photo = p
        uniform.addContribution(i, t, None)

    uniform.fillColumns()
    columns = mock_output_handler.getDataForProvider(uniform)
    assert len(columns.dtype.fields) == 8

    assert 'MY_A_A' in columns.dtype.fields
    assert 'MY_A_A_ERR' in columns.dtype.fields
    assert 'MY_B_B' in columns.dtype.fields
    assert 'MY_B_B_ERR' in columns.dtype.fields
    assert 'MY_C_C' in columns.dtype.fields
    assert 'MY_C_C_ERR' in columns.dtype.fields
    assert 'MY_C_A' in columns.dtype.fields
    assert 'MY_C_A_ERR' in columns.dtype.fields

    assert np.isclose(columns['MY_A_A'][0], catalog_photometry['FLUX_A'][0] * 0.75)
    assert np.isclose(columns['MY_B_B'][0], catalog_photometry['FLUX_B'][0] * 0.62745)
    assert np.isclose(columns['MY_C_C'][0], catalog_photometry['FLUX_C'][0] * 0.75)
    # In this case the output uniform photometry is C*, so the ratio is computed
    # as C* / A
    #   First  reference: 3.5187345 / (1*0.5626045) = 6.254366077768664
    #   Second reference: 2.6067620 / (2*0.9424200) = 1.3830150039260625
    #   So the mean ratio is 3.818690540847363
    assert np.isclose(columns['MY_C_A'][0], catalog_photometry['FLUX_A'][0] * 3.818690540847363)
